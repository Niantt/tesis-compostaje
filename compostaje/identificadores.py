"""Técnicas de identificación de fases (patrón Estrategia).

- IdentificadorPuntosCambio : detección de puntos de cambio (Etapa 2, D-07).
- IdentificadorDTW          : DTW 1-NN sobre ventanas de W días (Etapa 2, D-08).
- IdentificadorHibrido      : mejora de la Etapa 3. Los puntos de cambio proponen
                              los tramos y DTW decide la fase de cada tramo,
                              respetando el orden del proceso.
"""

from abc import ABC, abstractmethod

import numpy as np
import ruptures as rpt
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.metrics import cdist_dtw
from tslearn.utils import to_time_series_dataset

from .resultado import TRANSICIONES_VALIDAS, ResultadoFases
from .serie import FASES, SerieTemperatura

_ALGORITMOS = {"Dynp": rpt.Dynp, "Binseg": rpt.Binseg, "BottomUp": rpt.BottomUp}


class IdentificadorFases(ABC):
    """Interfaz común: recibe una serie diaria preprocesada y devuelve sus fases."""
    nombre = "abstracta"

    @abstractmethod
    def identificar(self, serie: SerieTemperatura) -> ResultadoFases:
        ...


# ---------------------------------------------------------------------------
# Puntos de cambio
# ---------------------------------------------------------------------------
def detectar_cambios(y: np.ndarray, n_cambios: int, costo: str = "linear",
                     algoritmo: str = "Binseg", min_dias: int = 2) -> list[int]:
    """Índices (día) donde comienza cada nuevo tramo."""
    n_cambios = min(n_cambios, max(len(y) // min_dias - 1, 0))
    if n_cambios == 0:
        return []
    senal = (np.column_stack([y, np.ones_like(y), np.arange(len(y))])
             if costo == "linear" else y.reshape(-1, 1))
    modelo = _ALGORITMOS[algoritmo](model=costo, min_size=min_dias, jump=1).fit(senal)
    return modelo.predict(n_bkps=n_cambios)[:-1]


def episodios_de_cambio(y: np.ndarray, umbral: float = 1.5, min_dias: int = 2) -> list[tuple[int, int]]:
    """Episodios de cambio rápido según la derivada discreta.

    La derivada discreta dT/dt = T[t+1] - T[t] (°C/día) mide la tasa de cambio. Un
    episodio es una racha de al menos `min_dias` días seguidos en que |dT/dt| >= umbral
    con el mismo signo (calentamiento o enfriamiento rápido). Devuelve [(inicio, fin)].
    """
    d = np.diff(y)
    signo = np.where(d >= umbral, 1, np.where(d <= -umbral, -1, 0))
    episodios, i = [], 0
    while i < len(signo):
        if signo[i] != 0:
            j = i
            while j + 1 < len(signo) and signo[j + 1] == signo[i]:
                j += 1
            if j - i + 1 >= min_dias:
                episodios.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return episodios


def n_cambios_por_derivada(y: np.ndarray, umbral: float = 1.5, minimo: int = 3) -> int:
    """Número de puntos de cambio sugerido por la derivada: cada episodio de cambio
    rápido aporta dos límites (su inicio y su fin). Nunca menos de `minimo`."""
    return max(minimo, 2 * len(episodios_de_cambio(y, umbral)))


class IdentificadorPuntosCambio(IdentificadorFases):
    """Busca 3 puntos de cambio y asigna las 4 fases en orden (Etapa 2)."""
    nombre = "Puntos de cambio"

    def __init__(self, costo: str = "linear", algoritmo: str = "Binseg", n_cambios: int = 3,
                 min_dias: int = 2):
        self.costo, self.algoritmo = costo, algoritmo
        self.n_cambios, self.min_dias = n_cambios, min_dias

    def identificar(self, serie):
        cambios = detectar_cambios(serie.temperaturas, self.n_cambios, self.costo,
                                   self.algoritmo, self.min_dias)
        idx = np.searchsorted(np.asarray(cambios), np.arange(len(serie)), side="right")
        fases = np.array(FASES)[np.minimum(idx, len(FASES) - 1)]
        return ResultadoFases(self.nombre, fases, serie.tiempos, dict(cambios=cambios))


# ---------------------------------------------------------------------------
# DTW
# ---------------------------------------------------------------------------
def ventanas(y: np.ndarray, w: int) -> np.ndarray:
    """Ventana de w días centrada en cada día (bordes rellenados con el extremo)."""
    h = w // 2
    yp = np.pad(y, h, mode="edge")
    return np.stack([yp[i:i + w] for i in range(len(y))])[..., None]


class ModeloFases:
    """Ventanas etiquetadas de cada fase (modelo 1-NN de DTW)."""

    def __init__(self, ventana: int = 21, senal: str = "celsius"):
        self.ventana, self.senal = ventana, senal
        self.X = None
        self.y = None

    def _senal(self, serie):
        return serie.metadatos["temp_norm"] if self.senal == "minmax" else serie.temperaturas

    def entrenar(self, series_etiquetadas: list[SerieTemperatura]) -> "ModeloFases":
        X, y = [], []
        for s in series_etiquetadas:
            m = np.array([f is not None for f in s.fase_ref])
            X.append(ventanas(self._senal(s), self.ventana)[m])
            y.append(s.fase_ref[m])
        self.X, self.y = np.vstack(X), np.concatenate(y).astype(object)
        return self

    @property
    def entrenado(self):
        return self.X is not None

    def guardar(self, ruta) -> None:
        """Guarda las ventanas etiquetadas (las usa la aplicación de escritorio)."""
        np.savez_compressed(ruta, X=self.X, y=self.y.astype(str), ventana=self.ventana,
                            senal=self.senal)

    @classmethod
    def cargar(cls, ruta) -> "ModeloFases":
        d = np.load(ruta, allow_pickle=False)
        m = cls(int(d["ventana"]), str(d["senal"]))
        m.X, m.y = d["X"], d["y"].astype(object)
        return m

    def distancias(self, serie: SerieTemperatura, radio: int) -> np.ndarray:
        """Matriz (días x fases): distancia DTW al vecino más cercano de cada fase."""
        D = cdist_dtw(ventanas(self._senal(serie), self.ventana), self.X,
                      global_constraint="sakoe_chiba", sakoe_chiba_radius=radio)
        return np.column_stack([D[:, self.y == f].min(axis=1) if np.any(self.y == f)
                                else np.full(len(D), np.inf) for f in FASES])


class ModeloFasesAdaptativo:
    """Modelo por fase con subcurvas de largo adaptado a la duración de cada fase.

    Retroalimentación de la Etapa 2: las fases iniciales duran mucho menos que 21
    días (mesófila de 2 a 7 días y termófila de 5 a 14 en las series etiquetadas),
    así que una ventana de 21 días mezcla varias fases. Aquí:
      - Mesófila y Termófila se representan con subcurvas cortas de `delta` días
        (2 o 3) que se deslizan dentro de la fase, día a día.
      - Enfriamiento y Maduración, que duran 29 a 58 días, con subcurvas de
        `ventana_larga` días, también deslizadas dentro de la fase.
    Las subcurvas de referencia no cruzan el límite de su fase. Si una fase es más
    corta que la subcurva, se usa la fase completa (DTW admite largos distintos).
    Para comparar fases con largos distintos, la distancia DTW se divide por la raíz
    del largo de la subcurva (queda en °C por día, como un RMS).
    """

    def __init__(self, delta: int = 3, ventana_larga: int = 21, senal: str = "celsius"):
        self.delta, self.ventana_larga, self.senal = delta, ventana_larga, senal
        self.largo = {"Mesófila": delta, "Termófila": delta,
                      "Enfriamiento": ventana_larga, "Maduración": ventana_larga}
        self.subcurvas = None       # {fase: dataset (n, largo, 1) con NaN de relleno}
        self.origen = None          # {fase: [(serie, día central)]}

    @property
    def ventana(self):              # compatibilidad con ModeloFases
        return self.ventana_larga

    def _senal(self, serie):
        return serie.metadatos["temp_norm"] if self.senal == "minmax" else serie.temperaturas

    def entrenar(self, series_etiquetadas: list[SerieTemperatura]) -> "ModeloFasesAdaptativo":
        sub = {f: [] for f in FASES}
        orig = {f: [] for f in FASES}
        for s in series_etiquetadas:
            y = self._senal(s)
            for fase, a, b in _tramos_referencia(s.fase_ref):
                L = self.largo[fase]
                if b - a <= L:
                    sub[fase].append(y[a:b])
                    orig[fase].append((s.nombre, (a + b - 1) // 2))
                else:
                    for i in range(a, b - L + 1):
                        sub[fase].append(y[i:i + L])
                        orig[fase].append((s.nombre, i + (L - 1) // 2))
        self.subcurvas = {f: to_time_series_dataset(v) for f, v in sub.items() if v}
        self.origen = orig
        return self

    @property
    def entrenado(self):
        return self.subcurvas is not None

    def ventanas_consulta(self, serie, fase) -> np.ndarray:
        """Subcurva de largo L(fase) alrededor de cada día de la serie."""
        L = self.largo[fase]
        y = np.pad(self._senal(serie), ((L - 1) // 2, L // 2), mode="edge")
        return np.stack([y[i:i + L] for i in range(len(serie))])[..., None]

    def distancias_por_subcurva(self, serie, radio) -> dict:
        """{fase: matriz (días x subcurvas de esa fase)} de distancias DTW normalizadas."""
        return {f: cdist_dtw(self.ventanas_consulta(serie, f), X, global_constraint="sakoe_chiba",
                             sakoe_chiba_radius=radio) / np.sqrt(self.largo[f])
                for f, X in self.subcurvas.items()}

    def distancias(self, serie: SerieTemperatura, radio: int) -> np.ndarray:
        """Matriz (días x fases): distancia a la subcurva más parecida de cada fase."""
        D = self.distancias_por_subcurva(serie, radio)
        return np.column_stack([D[f].min(axis=1) if f in D else np.full(len(serie), np.inf)
                                for f in FASES])

    def plantillas(self, max_iter: int = 10) -> dict:
        """Plantilla modelo de cada fase: promedio DBA de sus subcurvas."""
        return {f: dtw_barycenter_averaging(X, barycenter_size=self.largo[f], max_iter=max_iter)
                for f, X in self.subcurvas.items()}


def _tramos_referencia(fase_ref):
    """[(fase, inicio, fin)] de los tramos etiquetados de una serie."""
    tramos, ini = [], 0
    for i in range(1, len(fase_ref) + 1):
        if i == len(fase_ref) or fase_ref[i] != fase_ref[ini]:
            if fase_ref[ini] is not None:
                tramos.append((fase_ref[ini], ini, i))
            ini = i
    return tramos


class IdentificadorDTW(IdentificadorFases):
    """Asigna a cada día la fase de la ventana etiquetada más parecida (1-NN)."""
    nombre = "DTW"

    def __init__(self, modelo: ModeloFases, radio: int = 2):
        self.modelo, self.radio = modelo, radio

    def identificar(self, serie):
        D = self.modelo.distancias(serie, self.radio)
        fases = np.array(FASES, dtype=object)[D.argmin(axis=1)]
        return ResultadoFases(self.nombre, fases, serie.tiempos, dict(distancias=D))


# ---------------------------------------------------------------------------
# Híbrido (Etapa 3)
# ---------------------------------------------------------------------------
def decodificar_tramos(costo_tramo: np.ndarray, transiciones=TRANSICIONES_VALIDAS,
                       inicio=("Mesófila",)) -> list[int]:
    """Fase de menor costo total para cada tramo, con transiciones válidas.

    costo_tramo: matriz (tramos x fases). Dos tramos seguidos pueden tener la misma
    fase (se fusionan). Programación dinámica (Viterbi) sobre los tramos.
    """
    n, k = costo_tramo.shape
    permitido = np.array([[a == b or (FASES[a], FASES[b]) in transiciones for b in range(k)]
                          for a in range(k)])
    acumulado = np.full((n, k), np.inf)
    previo = np.zeros((n, k), dtype=int)
    for j in range(k):
        if FASES[j] in inicio:
            acumulado[0, j] = costo_tramo[0, j]
    for i in range(1, n):
        for j in range(k):
            cand = np.where(permitido[:, j], acumulado[i - 1], np.inf)
            previo[i, j] = int(np.argmin(cand))
            acumulado[i, j] = cand[previo[i, j]] + costo_tramo[i, j]
    camino = [int(np.argmin(acumulado[-1]))]
    for i in range(n - 1, 0, -1):
        camino.append(previo[i, camino[-1]])
    return camino[::-1]


class IdentificadorHibrido(IdentificadorFases):
    """Puntos de cambio proponen los tramos y DTW decide la fase de cada tramo.

    1. Sobresegmenta la curva con n_tramos - 1 puntos de cambio (más que 3, para
       poder representar recalentamientos).
    2. Para cada tramo, suma las distancias DTW de sus días a cada fase.
    3. Elige la secuencia de fases de menor costo que respete el orden del
       proceso (TRANSICIONES_VALIDAS), con programación dinámica.
    4. Regla térmica (opcional): un tramo solo puede ser Termófila si alcanza
       umbral_termofila (40 °C; Azim et al., 2018; Pérez-San Martín et al., 2025).
    Con restringir=False, cada tramo toma la fase de menor costo sin restricción
    (variante usada para medir el aporte de la restricción de orden).
    Con n_tramos="derivada", el número de puntos de cambio se calcula en cada curva
    a partir de la derivada discreta (n_cambios_por_derivada) en vez de fijarlo.
    """
    nombre = "Híbrido"

    def __init__(self, modelo: ModeloFases, radio: int = 2, n_tramos: int | str = 8,
                 costo: str = "linear", algoritmo: str = "Binseg", min_dias: int = 2,
                 restringir: bool = True, umbral_termofila: float | None = 40.0,
                 umbral_derivada: float = 1.5):
        self.modelo, self.radio = modelo, radio
        self.n_tramos, self.costo, self.algoritmo = n_tramos, costo, algoritmo
        self.min_dias, self.restringir = min_dias, restringir
        self.umbral_termofila, self.umbral_derivada = umbral_termofila, umbral_derivada

    def identificar(self, serie):
        n_cambios = (n_cambios_por_derivada(serie.temperaturas, self.umbral_derivada)
                     if self.n_tramos == "derivada" else self.n_tramos - 1)
        cambios = detectar_cambios(serie.temperaturas, n_cambios, self.costo,
                                   self.algoritmo, self.min_dias)
        bordes = [0] + list(cambios) + [len(serie)]
        D = self.modelo.distancias(serie, self.radio)
        costo_tramo = np.array([D[a:b].sum(axis=0) for a, b in zip(bordes[:-1], bordes[1:])])
        if self.umbral_termofila is not None:
            # se usa la curva interpolada sin suavizar: el suavizado rebaja los picos cortos
            y = serie.metadatos.get("temp_interp", serie.temperaturas)
            frio = np.array([y[a:b].max() < self.umbral_termofila
                             for a, b in zip(bordes[:-1], bordes[1:])])
            costo_tramo[frio, FASES.index("Termófila")] = np.inf
        if self.restringir:
            etiquetas = decodificar_tramos(costo_tramo)
        else:
            etiquetas = list(costo_tramo.argmin(axis=1))
        fases = np.empty(len(serie), dtype=object)
        for (a, b), j in zip(zip(bordes[:-1], bordes[1:]), etiquetas):
            fases[a:b] = FASES[j]
        return ResultadoFases(self.nombre, fases, serie.tiempos,
                              dict(cambios=cambios, distancias=D))

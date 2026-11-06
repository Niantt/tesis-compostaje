"""Clasificación causal de fases con plantillas por fase, características DTW y KNN.

Propuesta del profesor guía (retroalimentación de la Etapa 2):

1. Plantilla modelo de cada fase: promedio DBA de los tramos de esa fase en las
   curvas de referencia. Su largo es la duración típica (mediana) de la fase, no
   una ventana fija de 21 días.
2. Fragmentos causales: el fragmento del día t contiene las temperaturas de los
   días t-L+1 .. t (solo pasado y presente) y se etiqueta con la fase del día t.
   Al inicio de la curva, el fragmento es más corto (los días que existan).
3. Características DTW: para cada fragmento y cada plantilla, el costo DTW de
   subsecuencia (la mejor alineación del fragmento con alguna parte de la
   plantilla), dividido por la raíz del largo del fragmento. Con varios largos L
   se obtienen 4 características por largo.
4. KNN sobre esas características, entrenado con curvas distintas a las usadas
   para construir las plantillas.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.metrics import dtw_subsequence_path
from tslearn.utils import to_time_series_dataset

from .identificadores import IdentificadorFases, _tramos_referencia
from .resultado import ResultadoFases
from .serie import FASES, SerieTemperatura


def fragmentos_causales(y: np.ndarray, largo: int) -> list[np.ndarray]:
    """Fragmento de cada día t: y[max(0, t-largo+1) .. t]."""
    return [y[max(0, t - largo + 1):t + 1] for t in range(len(y))]


class PlantillasFase:
    """Plantilla modelo (DBA) de cada fase, con el largo típico de la fase."""

    def __init__(self, max_iter: int = 10):
        self.max_iter = max_iter
        self.plantillas = None
        self.duraciones = None

    def entrenar(self, curvas: list[SerieTemperatura]) -> "PlantillasFase":
        """`curvas` puede repetir una misma curva (copias del bootstrap)."""
        tramos = {f: [] for f in FASES}
        for c in curvas:
            for fase, a, b in _tramos_referencia(c.fase_ref):
                tramos[fase].append(c.temperaturas[a:b])
        self.duraciones = {f: int(np.median([len(t) for t in v])) for f, v in tramos.items() if v}
        self.plantillas = {}
        for f, v in tramos.items():
            if not v:
                continue
            X = to_time_series_dataset(v)
            if len(v) == 1:
                self.plantillas[f] = np.asarray(v[0], float)
            else:
                self.plantillas[f] = dtw_barycenter_averaging(
                    X, barycenter_size=max(self.duraciones[f], 2), max_iter=self.max_iter).ravel()
        return self

    def caracteristicas(self, y: np.ndarray, largos: tuple[int, ...]) -> np.ndarray:
        """Matriz (días x 4·len(largos)) de costos DTW de subsecuencia normalizados."""
        cols = []
        for L in largos:
            frags = fragmentos_causales(y, L)
            for f in FASES:
                T = self.plantillas[f]
                cols.append([dtw_subsequence_path(q, T)[1] / np.sqrt(len(q)) for q in frags])
        return np.array(cols).T


class ClasificadorKNNDTW:
    """KNN entrenado con las características DTW de un grupo de curvas."""

    def __init__(self, plantillas: PlantillasFase, largos=(3, 7, 14), k: int = 3):
        self.plantillas, self.largos, self.k = plantillas, tuple(largos), k
        self.escala = StandardScaler()
        self.knn = None

    def entrenar(self, curvas: list[SerieTemperatura]) -> "ClasificadorKNNDTW":
        X, y = [], []
        for c in curvas:
            m = np.array([f is not None for f in c.fase_ref])
            X.append(self.plantillas.caracteristicas(c.temperaturas, self.largos)[m])
            y.append(c.fase_ref[m])
        X, y = np.vstack(X), np.concatenate(y).astype(str)
        self.knn = KNeighborsClassifier(n_neighbors=min(self.k, len(y))).fit(self.escala.fit_transform(X), y)
        return self

    def predecir(self, serie: SerieTemperatura) -> np.ndarray:
        X = self.plantillas.caracteristicas(serie.temperaturas, self.largos)
        return self.knn.predict(self.escala.transform(X)).astype(object)


class IdentificadorKNNDTW(IdentificadorFases):
    nombre = "KNN-DTW (plantillas por fase)"

    def __init__(self, clasificador: ClasificadorKNNDTW):
        self.clasificador = clasificador

    def identificar(self, serie):
        return ResultadoFases(self.nombre, self.clasificador.predecir(serie), serie.tiempos)

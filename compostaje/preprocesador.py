"""Preprocesamiento de una serie: periodo diario, interpolación PCHIP y suavizado Savitzky-Golay."""

import numpy as np
from scipy.interpolate import PchipInterpolator

from . import filtros

from .serie import SerieTemperatura


class Preprocesador:
    """Lleva una serie cruda a la grilla diaria que usan las técnicas.

    Pasos (decisión D-06 de la Etapa 2):
      a) registros con varias mediciones por día (p. ej., horarios) -> promedio diario
      b) interpolación PCHIP sobre la grilla de 1 día (no genera picos artificiales)
      c) filtro de ruido, a elegir:
           "savgol"        Savitzky-Golay (ventana 7 días, orden 2; ventana 3 si < 30 días)
           "mediana_media" mediana móvil + media móvil (centradas, o causales con causal=True)
           "polinomio"     polinomio de grado 12 ajustado a cada curva
    La versión normalizada min-max queda en metadatos["temp_norm"].
    """

    FILTROS = ("savgol", "mediana_media", "polinomio")

    def __init__(self, periodo: int = 1, ventana_sg: int = 7, ventana_sg_corta: int = 3,
                 orden_sg: int = 2, umbral_corta: int = 30, filtro: str = "savgol",
                 ventana_mediana: int = 5, ventana_media: int = 3, causal: bool = False,
                 grado: int = 12):
        if filtro not in self.FILTROS:
            raise ValueError(f"Filtro desconocido: {filtro}")
        self.filtro, self.causal, self.grado = filtro, causal, grado
        self.ventana_mediana, self.ventana_media = ventana_mediana, ventana_media
        self.periodo = periodo
        self.ventana_sg = ventana_sg
        self.ventana_sg_corta = ventana_sg_corta
        self.orden_sg = orden_sg
        self.umbral_corta = umbral_corta

    def procesar(self, serie: SerieTemperatura) -> SerieTemperatura:
        t = serie.tiempos - serie.tiempos.min()
        y = serie.temperaturas
        ok = np.isfinite(t) & np.isfinite(y)
        t, y = t[ok], y[ok]
        if len(t) < 2:
            raise ValueError("La serie necesita al menos dos mediciones válidas")

        # a) agrupar por día (registro horario) o fusionar tiempos repetidos
        tipo = serie.metadatos.get("tipo_registro")
        if tipo is not None:
            horario = str(tipo).startswith("horario")
        else:   # archivo importado por el usuario: se detecta por el intervalo típico
            horario = len(t) > 2 and np.median(np.diff(np.sort(t))) < 0.5
        clave = np.floor(t / self.periodo) * self.periodo if horario else t
        x, inv = np.unique(clave, return_inverse=True)
        yx = np.bincount(inv, weights=y) / np.bincount(inv)
        if len(x) < 2:
            raise ValueError("La serie cubre menos de dos días")

        grilla = np.arange(0, np.floor(x.max()) + 1, self.periodo, dtype=float)

        # b) interpolación PCHIP
        y_int = PchipInterpolator(x, yx)(grilla)

        # c) filtro de ruido
        extra = {}
        if self.filtro == "savgol":
            y_sg = filtros.savitzky_golay(y_int, self.ventana_sg, self.orden_sg,
                                          self.ventana_sg_corta, self.umbral_corta)
        elif self.filtro == "mediana_media":
            y_sg = filtros.mediana_media(y_int, self.ventana_mediana, self.ventana_media, self.causal)
        else:
            y_sg, extra["grado_polinomio"] = filtros.polinomio(x, yx, grilla, self.grado)

        rango = y_sg.max() - y_sg.min()
        y_norm = (y_sg - y_sg.min()) / rango if rango > 0 else np.zeros_like(y_sg)

        meta = dict(serie.metadatos, temp_interp=y_int, temp_norm=y_norm,
                    es_medicion=np.isin(grilla, x), preprocesada=True,
                    filtro=self.filtro, **extra)
        return SerieTemperatura(serie.nombre, grilla, y_sg, metadatos=meta)


def asignar_fases_referencia(serie: SerieTemperatura, inicios: list[tuple[str, float]]) -> SerieTemperatura:
    """Agrega la fase de referencia a una serie diaria.

    inicios: [(fase, día de inicio)] con día 1 = primer día del registro,
    como vienen en la compilación de datos.
    """
    inicios = sorted(inicios, key=lambda p: p[1])
    dia = serie.tiempos + 1
    bordes = np.array([d for _, d in inicios])
    idx = np.searchsorted(bordes, dia, side="right") - 1
    fases = np.array([inicios[i][0] if i >= 0 else None for i in idx], dtype=object)
    serie.fase_ref = fases
    return serie

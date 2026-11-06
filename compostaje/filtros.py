"""Filtros de ruido para la curva diaria de temperatura.

- savitzky_golay : polinomios locales (decisión D-06 de la Etapa 2).
- mediana_media  : mediana móvil (quita picos aislados) seguida de media móvil
                   (suaviza lo que queda). Pedido del profesor guía en la
                   retroalimentación de la Etapa 2.
- polinomio      : un polinomio de grado 12 ajustado a toda la curva.

Los filtros centrados usan días anteriores y posteriores. La versión causal de
mediana_media usa solo el día actual y los anteriores: es la que corresponde
cuando cada fragmento debe usar únicamente temperaturas disponibles hasta ese día.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import savgol_filter


def savitzky_golay(y, ventana=7, orden=2, ventana_corta=3, umbral_corta=30):
    base = ventana if len(y) >= umbral_corta else ventana_corta
    v = min(base, len(y) if len(y) % 2 else len(y) - 1)
    return savgol_filter(y, v, min(orden, v - 1)) if v >= 3 else np.asarray(y, float).copy()


def mediana_media(y, ventana_mediana=5, ventana_media=3, causal=False):
    """En la versión centrada, los bordes se rellenan repitiendo el primer y el último
    valor; así el primer día conserva su temperatura y no hereda la de días posteriores."""
    y = np.asarray(y, float)
    if not causal:
        m = median_filter(y, size=ventana_mediana, mode="nearest")
        return uniform_filter1d(m, size=ventana_media, mode="nearest")
    s = pd.Series(y)
    s = s.rolling(ventana_mediana, min_periods=1).median()
    s = s.rolling(ventana_media, min_periods=1).mean()
    return s.to_numpy()


def polinomio(t_medido, y_medido, t_grilla, grado=12):
    """Ajuste por mínimos cuadrados. El grado se limita si hay pocas mediciones,
    porque con n puntos un polinomio de grado n-1 o más solo los interpola."""
    g = int(min(grado, max(len(t_medido) - 3, 1)))
    p = np.polynomial.Polynomial.fit(t_medido, y_medido, g)   # dominio reescalado: estable
    return p(t_grilla), g

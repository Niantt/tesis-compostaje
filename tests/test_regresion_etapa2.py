"""Prueba de regresión: el paquete reproduce los resultados de los scripts de la Etapa 2.

Requiere el Excel en la raíz y haber ejecutado etapa2/preprocesamiento.py,
changepoint.py y dtw_fases.py (sus salidas quedan en etapa2/salida*/). Si no
están, la prueba se omite.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compostaje import IdentificadorDTW, IdentificadorPuntosCambio, ModeloFases
from compostaje.compilacion import cargar_conjunto

RAIZ = Path(__file__).resolve().parents[1]
XLSX = RAIZ / "Compilacion_temperatura_compost.xlsx"
E2 = RAIZ / "etapa2"
pytestmark = pytest.mark.skipif(not (XLSX.exists() and (E2 / "salida_dtw").exists()),
                                reason="faltan los datos o las salidas de la Etapa 2")


@pytest.fixture(scope="module")
def series():
    return {s.nombre: s for s in cargar_conjunto(XLSX)[0]}


def test_mismo_conjunto_y_preprocesamiento(series):
    diario = pd.read_csv(E2 / "salida" / "datos_diarios_preprocesados.csv")
    assert set(series) == set(diario.serie_id) and len(series) == 33
    for sid, g in diario.groupby("serie_id"):
        assert np.allclose(series[sid].temperaturas, g.temp_suav_c, atol=1e-3)
        ref = g.fase_ref.where(g.fase_ref.notna(), None).to_numpy(dtype=object)
        if series[sid].fase_ref is not None:
            assert list(series[sid].fase_ref) == list(ref)


def test_mismos_puntos_de_cambio(series):
    cp = pd.read_csv(E2 / "salida_changepoint" / "fases_detectadas_por_dia.csv")
    tec = IdentificadorPuntosCambio()
    for sid, g in cp.groupby("serie_id"):
        assert list(tec.identificar(series[sid]).fase_por_dia) == list(g.fase_detectada)


def test_mismas_fases_dtw_en_series_sin_etiquetas(series):
    dtw = pd.read_csv(E2 / "salida_dtw" / "fases_dtw_por_dia.csv")
    etiquetadas = [s for s in series.values() if s.etiquetada]
    tec = IdentificadorDTW(ModeloFases(21).entrenar(etiquetadas))
    for sid, g in dtw.groupby("serie_id"):
        if not series[sid].etiquetada:
            assert list(tec.identificar(series[sid]).fase_por_dia) == list(g.fase_dtw)

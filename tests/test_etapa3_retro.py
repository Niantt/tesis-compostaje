"""Pruebas de lo agregado tras la retroalimentación de la Etapa 2."""

import numpy as np
import pytest
from conftest import curva_sintetica, serie_etiquetada

from compostaje import FASES, Preprocesador
from compostaje.filtros import mediana_media, polinomio
from compostaje.identificadores import episodios_de_cambio, n_cambios_por_derivada
from compostaje.knn_dtw import ClasificadorKNNDTW, PlantillasFase, fragmentos_causales


def test_mediana_quita_pico_aislado():
    y = np.full(15, 30.0)
    y[7] = 80                                         # error de medición
    assert mediana_media(y, 5, 3).max() < 31


def test_mediana_centrada_respeta_el_borde():
    y = np.array([19, 37, 51, 60, 66, 64, 60], float)
    assert mediana_media(y, 5, 3)[0] < 30             # no hereda el calor de días posteriores


def test_filtro_causal_no_usa_el_futuro():
    y = np.array([20, 21, 22, 23, 24, 60, 61, 62], float)
    f1 = mediana_media(y, 5, 3, causal=True)
    y2 = y.copy()
    y2[5:] = 0                                        # cambiar el futuro no altera el pasado
    f2 = mediana_media(y2, 5, 3, causal=True)
    assert np.allclose(f1[:5], f2[:5])


def test_polinomio_limita_grado_con_pocas_mediciones():
    t = np.arange(8.0)
    _, g = polinomio(t, np.sin(t), np.arange(8.0), grado=12)
    assert g < 8


@pytest.mark.parametrize("filtro", Preprocesador.FILTROS)
def test_preprocesador_con_cada_filtro(filtro):
    s = Preprocesador(filtro=filtro).procesar(curva_sintetica()[0])
    assert s.es_periodica() and np.all(np.isfinite(s.temperaturas))


def test_derivada_detecta_subida_y_bajada():
    y = np.concatenate([np.linspace(20, 60, 6), np.full(10, 60), np.linspace(60, 25, 15), np.full(20, 25)])
    ep = episodios_de_cambio(y, umbral=1.5)
    assert len(ep) == 2
    assert n_cambios_por_derivada(y, 1.5) == 4
    assert n_cambios_por_derivada(np.full(30, 25.0), 1.5) == 3      # mínimo


def test_fragmentos_causales():
    f = fragmentos_causales(np.arange(10.0), 3)
    assert len(f) == 10 and list(f[0]) == [0] and list(f[5]) == [3, 4, 5]


def test_plantillas_tienen_largo_de_la_fase(etiquetadas):
    p = PlantillasFase().entrenar(etiquetadas)
    for f in FASES:
        assert len(p.plantillas[f]) == p.duraciones[f]
    assert p.duraciones["Mesófila"] < p.duraciones["Maduración"]


def test_knn_dtw_clasifica_curva_nueva(etiquetadas):
    p = PlantillasFase().entrenar(etiquetadas[:2])
    c = ClasificadorKNNDTW(p, largos=(3, 7)).entrenar(etiquetadas[2:])
    s = serie_etiquetada(nombre="N", semilla=11)
    assert np.mean(c.predecir(s) == s.fase_ref) > 0.6

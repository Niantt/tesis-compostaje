"""Datos sintéticos para las pruebas: curvas con las cuatro fases y límites conocidos."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compostaje import Preprocesador, SerieTemperatura, asignar_fases_referencia  # noqa: E402


def curva_sintetica(nombre="SINT", dias=90, t_ini=20.0, t_max=60.0, t_mad=22.0,
                    inicio_term=5, inicio_enfr=20, inicio_mad=45, ruido=0.8, semilla=0,
                    recalentamiento=None):
    """Curva diaria por tramos lineales + ruido. Devuelve (serie cruda, inicios de fase)."""
    rng = np.random.default_rng(semilla)
    t = np.arange(dias, dtype=float)
    y = np.interp(t, [0, inicio_term, inicio_term + 6, inicio_enfr, inicio_mad, dias - 1],
                  [t_ini, 42, t_max, 45, t_mad + 3, t_mad])
    if recalentamiento is not None:                       # pico de volteo durante el enfriamiento
        d0 = recalentamiento
        y += np.interp(t, [d0, d0 + 3, d0 + 8], [0, 15, 0])
    y += rng.normal(0, ruido, dias)
    inicios = [("Mesófila", 1), ("Termófila", inicio_term + 1),
               ("Enfriamiento", inicio_enfr + 1), ("Maduración", inicio_mad + 1)]
    return SerieTemperatura(nombre, t, y), inicios


def serie_etiquetada(**kw):
    cruda, inicios = curva_sintetica(**kw)
    return asignar_fases_referencia(Preprocesador().procesar(cruda), inicios)


@pytest.fixture
def etiquetadas():
    return [serie_etiquetada(nombre=f"E{i}", semilla=i, inicio_term=4 + i, inicio_enfr=18 + 2 * i,
                             inicio_mad=40 + 3 * i, t_mad=18 + 3 * i) for i in range(4)]

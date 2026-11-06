"""Pruebas de integración de la aplicación (Vista + Controlador + Modelo), sin pantalla."""

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from app.controlador import RUTA_MODELO, ControladorPrincipal  # noqa: E402
from app.vista import VentanaPrincipal  # noqa: E402
from conftest import curva_sintetica  # noqa: E402


@pytest.fixture
def controlador(qtbot, monkeypatch):
    c = ControladorPrincipal(VentanaPrincipal)
    qtbot.addWidget(c.vista)
    errores = []
    monkeypatch.setattr(c.vista, "mostrar_error", errores.append)
    c.errores = errores
    return c


@pytest.fixture
def csv_valido(tmp_path):
    s, _ = curva_sintetica()
    ruta = tmp_path / "pila.csv"
    pd.DataFrame({"Día": s.tiempos, "Temperatura": s.temperaturas}).to_csv(ruta, index=False)
    return ruta


def test_tecnicas_disponibles(controlador):
    esperadas = 3 if RUTA_MODELO.exists() else 1
    assert controlador.vista.combo_tecnica.count() == esperadas


def test_flujo_completo(controlador, csv_valido, tmp_path):
    vista = controlador.vista
    assert not vista.accion_identificar.isEnabled()
    controlador.abrir_archivo(str(csv_valido))
    assert vista.accion_identificar.isEnabled() and not vista.accion_exportar.isEnabled()
    for i in range(vista.combo_tecnica.count()):
        tecnica = vista.combo_tecnica.itemText(i)
        controlador.identificar_fases(tecnica)
        if tecnica != "DTW":            # DTW solo no garantiza el orden (limitación de la Etapa 2)
            assert controlador.resultado.es_coherente()
        assert vista.lbl_fase.text() == controlador.resultado.fase_actual()
    controlador.exportar(str(tmp_path / "salida.csv"))
    assert len(pd.read_csv(tmp_path / "salida.csv")) == len(controlador.serie_actual)
    assert controlador.errores == []


def test_archivo_malformado_muestra_error(controlador, tmp_path):
    ruta = tmp_path / "malo.csv"
    ruta.write_text("hola\nmundo\n", encoding="utf-8")
    controlador.abrir_archivo(str(ruta))
    assert controlador.errores and controlador.serie_actual is None


def test_serie_invalida_no_se_carga(controlador, tmp_path):
    ruta = tmp_path / "corta.csv"
    ruta.write_text("Día,Temperatura\n0,20\n1,25\n2,30\n", encoding="utf-8")
    controlador.abrir_archivo(str(ruta))
    assert controlador.errores and controlador.serie_actual is None

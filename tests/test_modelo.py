"""Pruebas unitarias de la capa Modelo."""

import numpy as np
import pandas as pd
import pytest
from conftest import curva_sintetica, serie_etiquetada

from compostaje import (FASES, ErrorImportacion, ExportadorReporte, IdentificadorDTW,
                        IdentificadorHibrido, IdentificadorPuntosCambio, ImportadorDatos,
                        ModeloFases, Preprocesador, ResultadoFases, SerieTemperatura,
                        ValidadorDatos)
from compostaje.identificadores import decodificar_tramos


# --- SerieTemperatura / ResultadoFases --------------------------------------
def test_serie_largos_distintos():
    with pytest.raises(ValueError):
        SerieTemperatura("x", [0, 1, 2], [20, 21])


def test_serie_periodica_y_duracion():
    s = SerieTemperatura("x", [0, 1, 2, 3], [20, 30, 40, 35])
    assert s.es_periodica() and s.duracion() == 3
    assert not SerieTemperatura("x", [0, 1, 3], [1, 2, 3]).es_periodica()


def test_resultado_segmentos_y_fase_actual():
    r = ResultadoFases("t", ["Mesófila"] * 2 + ["Termófila"] * 3 + ["Enfriamiento"], np.arange(6))
    assert r.segmentos() == [("Mesófila", 0, 2), ("Termófila", 2, 5), ("Enfriamiento", 5, 6)]
    assert r.fase_actual() == "Enfriamiento"
    assert r.duracion_por_fase()["Termófila"] == 3


@pytest.mark.parametrize("secuencia,esperado", [
    (["Mesófila", "Termófila", "Enfriamiento", "Maduración"], True),
    (["Mesófila", "Termófila", "Enfriamiento", "Termófila", "Enfriamiento"], True),   # recalentamiento
    (["Maduración", "Mesófila"], False),                                               # parte mal
    (["Mesófila", "Termófila", "Mesófila"], False),                                    # retrocede
    (["Mesófila", "Termófila", "Enfriamiento", "Maduración", "Termófila"], False),
])
def test_coherencia(secuencia, esperado):
    r = ResultadoFases("t", secuencia, np.arange(len(secuencia)))
    assert r.es_coherente() == esperado


# --- Preprocesador ------------------------------------------------------------
def test_preprocesador_grilla_diaria_desde_irregular():
    s = SerieTemperatura("x", [0, 2, 5, 9, 14, 20], [20, 35, 55, 50, 35, 25])
    p = Preprocesador().procesar(s)
    assert p.es_periodica() and len(p) == 21
    assert np.all(np.isfinite(p.temperaturas))
    assert p.metadatos["es_medicion"].sum() == 6


def test_preprocesador_promedia_registro_horario():
    t = np.arange(0, 10, 1 / 24)
    s = SerieTemperatura("x", t, 30 + np.sin(2 * np.pi * t), metadatos=dict(tipo_registro="horario (sensor)"))
    p = Preprocesador().procesar(s)
    assert len(p) == 10
    assert np.allclose(p.metadatos["temp_interp"], 30, atol=0.05)   # el ciclo diario se promedia


def test_preprocesador_normalizacion():
    p = Preprocesador().procesar(curva_sintetica()[0])
    n = p.metadatos["temp_norm"]
    assert n.min() == pytest.approx(0) and n.max() == pytest.approx(1)


def test_asigna_fases_referencia():
    s = serie_etiquetada(inicio_term=5, inicio_enfr=20, inicio_mad=45)
    assert s.fase_ref[0] == "Mesófila" and s.fase_ref[5] == "Termófila"
    assert s.fase_ref[19] == "Termófila" and s.fase_ref[20] == "Enfriamiento"
    assert s.fase_ref[-1] == "Maduración"


# --- Técnicas -----------------------------------------------------------------
def _error_limites(resultado, serie):
    def ini(f):
        return [int(np.argmax(f == x)) for x in FASES[1:]]
    return np.abs(np.array(ini(resultado.fase_por_dia)) - np.array(ini(serie.fase_ref)))


def test_puntos_de_cambio_encuentra_limites(etiquetadas):
    s = etiquetadas[0]
    r = IdentificadorPuntosCambio().identificar(s)
    assert list(dict.fromkeys(r.fase_por_dia)) == FASES        # las 4 fases, en orden
    assert _error_limites(r, s).mean() < 6


def test_dtw_clasifica_serie_nueva(etiquetadas):
    modelo = ModeloFases(ventana=21).entrenar(etiquetadas[1:])
    r = IdentificadorDTW(modelo).identificar(etiquetadas[0])
    assert np.mean(r.fase_por_dia == etiquetadas[0].fase_ref) > 0.75


def test_hibrido_es_coherente_y_preciso(etiquetadas):
    modelo = ModeloFases(ventana=21).entrenar(etiquetadas[1:])
    r = IdentificadorHibrido(modelo).identificar(etiquetadas[0])
    assert r.es_coherente()
    assert np.mean(r.fase_por_dia == etiquetadas[0].fase_ref) > 0.75


def test_hibrido_reconoce_recalentamiento(etiquetadas):
    modelo = ModeloFases(ventana=21).entrenar(etiquetadas)
    s = serie_etiquetada(nombre="R", recalentamiento=30, semilla=9)
    r = IdentificadorHibrido(modelo).identificar(s)
    orden = [t[0] for t in r.segmentos()]
    assert r.es_coherente()
    assert orden.count("Termófila") >= 1 and "Maduración" in orden


def test_decodificacion_respeta_orden():
    # El costo mínimo sin restricción sería [Maduración, Mesófila]; con restricción, debe partir en Mesófila
    costo = np.array([[5, 9, 9, 1], [1, 9, 9, 9], [9, 1, 9, 9], [9, 9, 1, 9]], dtype=float)
    camino = [FASES[j] for j in decodificar_tramos(costo)]
    assert camino[0] == "Mesófila"
    assert camino == ["Mesófila", "Mesófila", "Termófila", "Enfriamiento"]


# --- Importador / Validador / Exportador ----------------------------------------
def test_importa_csv_con_fechas_y_coma_decimal(tmp_path):
    f = tmp_path / "pila.csv"
    f.write_text("Fecha;Temperatura (°C)\n01-03-2025;22,5\n03-03-2025;41,0\n02-03-2025;--\n06-03-2025;55\n",
                 encoding="utf-8")
    s = ImportadorDatos().leer(f)
    assert list(s.tiempos) == [0, 2, 5]
    assert list(s.temperaturas) == [22.5, 41.0, 55.0]
    assert s.metadatos["filas_descartadas"] == 1


def test_importa_xlsx_con_dias(tmp_path):
    f = tmp_path / "pila.xlsx"
    pd.DataFrame({"Día": [0, 1, 2, 3], "Temp": [20, 30, 45, 50]}).to_excel(f, index=False)
    s = ImportadorDatos().leer(f)
    assert len(s) == 4 and s.temperaturas[-1] == 50


@pytest.mark.parametrize("contenido", ["", "solo_una_columna\n1\n2\n", "a,b\nx,y\nz,w\n"])
def test_importa_archivos_malformados(tmp_path, contenido):
    f = tmp_path / "malo.csv"
    f.write_text(contenido, encoding="utf-8")
    with pytest.raises(ErrorImportacion):
        ImportadorDatos().leer(f)


def test_formato_no_soportado(tmp_path):
    with pytest.raises(ErrorImportacion):
        ImportadorDatos().leer(tmp_path / "datos.txt")


def test_validador_detecta_problemas():
    v = ValidadorDatos()
    s = SerieTemperatura("x", [0, 1, 2, 15, 16], [45, 50, 200, 30, 25])
    niveles = {a.nivel for a in v.validar(s)}
    assert v.hay_errores(v.validar(s)) and "advertencia" in niveles
    assert not v.hay_errores(v.validar(curva_sintetica()[0]))


def test_exporta_csv(tmp_path, etiquetadas):
    s = etiquetadas[0]
    r = IdentificadorPuntosCambio().identificar(s)
    ExportadorReporte().exportar_csv(s, r, tmp_path / "r.csv")
    df = pd.read_csv(tmp_path / "r.csv")
    assert len(df) == len(s) and set(df["fase"]) == set(FASES)

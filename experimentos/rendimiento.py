"""
rendimiento.py — Tesis INFO-1197, Etapa 3
Pruebas de rendimiento y estrés del flujo completo (importar → validar →
preprocesar → identificar) con registros sintéticos de largo creciente.

  - RNF-03: una serie de 90 días debe procesarse en pocos segundos.
  - Estrés: registros de hasta 2 años, registro horario de un año (8.760 filas)
    y archivos con filas corruptas.
  - Arranque en frío: la primera llamada a DTW incluye la compilación de tslearn.

Uso: python experimentos/rendimiento.py [--salida salida_etapa3]
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "tests"))
from compostaje import (IdentificadorDTW, IdentificadorHibrido, IdentificadorPuntosCambio,  # noqa: E402
                        ImportadorDatos, ModeloFases, Preprocesador, ValidadorDatos)
from conftest import curva_sintetica  # noqa: E402

MODELO = RAIZ / "app" / "recursos" / "modelo_fases.npz"
REPETICIONES = 3


def escribir_csv(ruta, dias, por_dia=1, corruptas=0, semilla=0):
    """Registro con fechas; por_dia=24 simula un sensor horario."""
    s, _ = curva_sintetica(dias=dias, inicio_term=5, inicio_enfr=int(dias * 0.25),
                           inicio_mad=int(dias * 0.55), semilla=semilla)
    t = np.arange(0, dias, 1 / por_dia)
    y = np.interp(t, s.tiempos, s.temperaturas) + np.random.default_rng(semilla).normal(0, 0.3, len(t))
    df = pd.DataFrame({"Fecha": pd.Timestamp("2025-04-01") + pd.to_timedelta(t, unit="D"),
                       "Temperatura": y.round(2).astype(str)})
    if corruptas:
        idx = np.random.default_rng(1).choice(len(df), corruptas, replace=False)
        df.loc[idx, "Temperatura"] = "--"
    df.to_csv(ruta, index=False)


def medir(ruta, tecnicas):
    filas = []
    for nombre, tec in tecnicas.items():
        tiempos = []
        for _ in range(REPETICIONES):
            t0 = time.perf_counter()
            cruda = ImportadorDatos().leer(ruta)
            ValidadorDatos().validar(cruda)
            serie = Preprocesador().procesar(cruda)
            r = tec.identificar(serie)
            tiempos.append(time.perf_counter() - t0)
        filas.append(dict(tecnica=nombre, filas=len(cruda), dias=len(serie),
                          segundos=np.median(tiempos), coherente=r.es_coherente()))
    return filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="salida_etapa3")
    out = Path(ap.parse_args().salida)
    out.mkdir(exist_ok=True)
    modelo = ModeloFases.cargar(MODELO)

    # arranque en frío (primera llamada en el proceso)
    s = Preprocesador().procesar(curva_sintetica(dias=90)[0])
    t0 = time.perf_counter()
    IdentificadorHibrido(modelo).identificar(s)
    frio = time.perf_counter() - t0
    print(f"Primera identificación (arranque en frío): {frio:.2f} s")

    tecnicas = {"Puntos de cambio": IdentificadorPuntosCambio(),
                "DTW": IdentificadorDTW(modelo),
                "Híbrido": IdentificadorHibrido(modelo, n_tramos="derivada", umbral_derivada=1.0)}
    casos = [("30 días, diario", 30, 1, 0), ("90 días, diario", 90, 1, 0),
             ("180 días, diario", 180, 1, 0), ("1 año, diario", 365, 1, 0),
             ("2 años, diario", 730, 1, 0), ("90 días, horario", 90, 24, 0),
             ("1 año, horario", 365, 24, 0), ("90 días, horario, 5 % corrupto", 90, 24, 108)]
    filas = []
    with tempfile.TemporaryDirectory() as tmp:
        for nombre, dias, por_dia, corr in casos:
            ruta = Path(tmp) / "registro.csv"
            escribir_csv(ruta, dias, por_dia, corr)
            for f in medir(ruta, tecnicas):
                filas.append(dict(caso=nombre, **f))
                print(f"  {nombre:<32} {f['tecnica']:<17} {f['filas']:>6} filas  "
                      f"{f['segundos']:6.2f} s  coherente={f['coherente']}")
    df = pd.DataFrame(filas)
    df.to_csv(out / "rendimiento.csv", index=False)
    tabla = df.pivot_table(index=["caso", "filas"], columns="tecnica", values="segundos",
                           sort=False).reset_index()
    tabla.to_csv(out / "rendimiento_tabla.csv", index=False)
    (out / "arranque_en_frio.txt").write_text(f"{frio:.2f}\n")
    print("\n", tabla.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

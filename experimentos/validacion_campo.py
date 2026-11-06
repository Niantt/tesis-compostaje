"""
validacion_campo.py — Tesis INFO-1197, Etapa 3
Validación preliminar en condiciones reales: compara las fases identificadas en
las pilas de campo con las anotaciones de diagnóstico registradas en terreno
(Pila 15 y Pila 11, campaña 2025). Estas pilas no se usan para entrenar.

Salidas (salida_etapa3/):
  validacion_campo.csv              inicio de cada fase según cada técnica (fechas)
  figuras/validacion_campo.png      curva, fases del software y anotaciones de campo

Uso: python experimentos/validacion_campo.py [ruta.xlsx]
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "experimentos"))
from compostaje import COLORES, FASES, ModeloFases  # noqa: E402
from compostaje.compilacion import cargar_conjunto, cargar_todo  # noqa: E402
from evaluar_tecnicas import FINAL, VENTANA, tecnicas  # noqa: E402

warnings.filterwarnings("ignore")
PILAS = ["PILA_15", "PILA_11"]
MOSTRAR = ["Puntos de cambio", "DTW", FINAL]
MESES = {"ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6, "jul": 7,
         "ago": 8, "sep": 9, "oct": 10, "nov": 11, "dic": 12}


def fecha_nota(texto, anio):
    """Fecha de inicio de una anotación: '2025-04-07' o rangos como '23 Abril a 2 Mayo'."""
    f = pd.to_datetime(texto, errors="coerce")
    if pd.notna(f):
        return f
    import re
    m = re.match(r"\s*(\d{1,2})\D+?(?:([A-Za-z]+)\s+)?a\s+\d{1,2}\s*([A-Za-z]+)", texto)
    if not m:
        return pd.NaT
    mes = MESES.get((m.group(2) or m.group(3)).lower()[:3])
    return pd.Timestamp(anio, mes, int(m.group(1))) if mes else pd.NaT


def main():
    ruta = sys.argv[1] if len(sys.argv) > 1 else "Compilacion_temperatura_compost.xlsx"
    out = Path("salida_etapa3")
    (out / "figuras").mkdir(parents=True, exist_ok=True)
    crudo, notas, _ = cargar_todo(ruta)
    series, _, _ = cargar_conjunto(ruta)
    por_id = {s.nombre: s for s in series}
    modelo = ModeloFases(VENTANA).entrenar([s for s in series if s.etiquetada])
    tec = tecnicas(modelo)

    filas = []
    fig, axs = plt.subplots(len(PILAS), len(MOSTRAR), figsize=(16, 4.2 * len(PILAS)), squeeze=False)
    for i, sid in enumerate(PILAS):
        s = por_id[sid]
        c = crudo[crudo.serie_id == sid].sort_values("fecha")
        t0 = c["fecha"].min()
        fechas = t0 + pd.to_timedelta(s.tiempos, unit="D")
        n = notas[notas.serie_id == sid]
        for j, nombre in enumerate(MOSTRAR):
            r = tec[nombre].identificar(s)
            fila = dict(pila=sid, tecnica=nombre, coherente=r.es_coherente())
            for f in FASES[1:]:
                idx = np.flatnonzero(r.fase_por_dia == f)
                fila[f"inicio_{f.lower()}"] = (t0 + pd.Timedelta(days=int(idx[0]))).date() if len(idx) else None
            filas.append(fila)
            ax = axs[i, j]
            for fase, a, b in r.segmentos():
                ax.axvspan(t0 + pd.Timedelta(days=a), t0 + pd.Timedelta(days=b),
                           color=COLORES[fase], alpha=0.6, lw=0)
            ax.plot(c["fecha"], c["temp_c"], "o", color="0.35", ms=4)
            ax.plot(fechas, s.temperaturas, color="k", lw=1.2)
            for k, nota in enumerate(n.itertuples()):
                f = fecha_nota(nota.fecha_o_rango, t0.year)
                if pd.notna(f):
                    ax.axvline(f, color="navy", ls="--", lw=0.9)
                    ax.text(f, 71 - 4.3 * (k % 7), f" {nota.diagnostico[:26]}", fontsize=6.5,
                            color="navy", va="top")
            ax.set_ylim(10, 73)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
            ax.tick_params(labelsize=8)
            if i == 0:
                ax.set_title(nombre, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{sid}\nTemperatura (°C)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORES[f], alpha=0.6) for f in FASES]
    handles.append(plt.Line2D([], [], color="navy", ls="--"))
    fig.legend(handles, FASES + ["Anotación de campo"], loc="lower center", ncol=5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "figuras" / "validacion_campo.png", dpi=150)
    plt.close(fig)

    tabla = pd.DataFrame(filas)
    tabla.to_csv(out / "validacion_campo.csv", index=False)
    print(tabla.to_string(index=False))
    print("\nAnotaciones:\n", notas[notas.serie_id.isin(PILAS)].to_string(index=False))


if __name__ == "__main__":
    main()

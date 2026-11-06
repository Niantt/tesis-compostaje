"""
plantillas_modelo.py — Tesis INFO-1197, Etapa 3
Plantilla modelo de cada fase con el largo típico de la fase (no una ventana fija
de 21 días) y su comparación con los tramos de las 5 series etiquetadas.

  - La plantilla de cada fase es el promedio DBA de los tramos de esa fase, con
    largo igual a la duración mediana de la fase (PlantillasFase).
  - Para comparar sin usar la propia serie, las plantillas se construyen dejando
    fuera la serie evaluada (5 conjuntos de plantillas).
  - Cada tramo etiquetado de la serie se compara con las 4 plantillas mediante DTW
    (distancia dividida por la raíz del largo del tramo). Se registra la plantilla
    más cercana: si coincide con la fase del tramo, la plantilla representa bien
    esa fase en esa serie.

Salidas: plantillas_modelo_vs_series.csv, figuras/plantillas_modelo.png,
         figuras/plantillas_modelo_vs_series.png

Uso: python experimentos/plantillas_modelo.py [ruta.xlsx]
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tslearn.metrics import dtw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compostaje import FASES  # noqa: E402
from compostaje.compilacion import cargar_conjunto  # noqa: E402
from compostaje.identificadores import _tramos_referencia  # noqa: E402
from compostaje.knn_dtw import PlantillasFase  # noqa: E402

warnings.filterwarnings("ignore")
LINEAS = {"Mesófila": "#3182bd", "Termófila": "#de2d26", "Enfriamiento": "#e6550d", "Maduración": "#31a354"}
CORTO = {"DOCTORAL_PILA_1": "Doct. 1", "DOCTORAL_PILA_2": "Doct. 2", "DOCTORAL_PILA_3": "Doct. 3",
         "PENAILILLO_C1_SH": "Peñ. C1", "PENAILILLO_C3_CW": "Peñ. C3"}


def main():
    ruta = sys.argv[1] if len(sys.argv) > 1 else "Compilacion_temperatura_compost.xlsx"
    out = Path("salida_etapa3")
    (out / "figuras").mkdir(parents=True, exist_ok=True)
    series, _, _ = cargar_conjunto(ruta)
    et = [s for s in series if s.etiquetada]

    # plantillas con todas las series (figura)
    todas = PlantillasFase().entrenar(et)
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.4), sharey=True)
    for ax, f in zip(axs, FASES):
        for s in et:
            for fase, a, b in _tramos_referencia(s.fase_ref):
                if fase == f:
                    ax.plot(np.arange(b - a) + 1, s.temperaturas[a:b], color="0.7", lw=0.9)
        T = todas.plantillas[f]
        ax.plot(np.arange(len(T)) + 1, T, color=LINEAS[f], lw=2.6)
        ax.set_title(f"{f}: plantilla de {len(T)} días", fontsize=10)
        ax.set_xlabel("Día de la fase")
    axs[0].set_ylabel("Temperatura (°C)")
    fig.tight_layout()
    fig.savefig(out / "figuras" / "plantillas_modelo.png", dpi=150)
    plt.close(fig)

    # comparación dejando la serie fuera
    filas = []
    for s in et:
        P = PlantillasFase().entrenar([o for o in et if o is not s])
        for fase, a, b in _tramos_referencia(s.fase_ref):
            q = s.temperaturas[a:b]
            d = {f: dtw(q, P.plantillas[f]) / np.sqrt(len(q)) for f in FASES}
            cerca = min(d, key=d.get)
            filas.append(dict(serie_id=s.nombre, fase=fase, dias=b - a,
                              largo_plantilla=len(P.plantillas[fase]), plantilla_mas_cercana=cerca,
                              acierto=cerca == fase, **{f"dist_{f.lower()}": round(v, 2) for f, v in d.items()}))
    tabla = pd.DataFrame(filas)
    tabla.to_csv(out / "plantillas_modelo_vs_series.csv", index=False)

    # figura: matriz serie x fase con la distancia a la plantilla propia y la más cercana
    M = tabla.pivot(index="serie_id", columns="fase", values="plantilla_mas_cercana").loc[
        [s.nombre for s in et], FASES]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for i, sid in enumerate(M.index):
        for j, f in enumerate(FASES):
            ok = M.loc[sid, f] == f
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color="#c7e9c0" if ok else "#fcbba1"))
            ax.text(j + 0.5, i + 0.5, M.loc[sid, f], ha="center", va="center", fontsize=8)
    ax.set_xlim(0, 4)
    ax.set_ylim(len(M), 0)
    ax.set_xticks(np.arange(4) + 0.5, [f"Tramo {f.lower()}" for f in FASES], fontsize=8)
    ax.set_yticks(np.arange(len(M)) + 0.5, [CORTO[s] for s in M.index], fontsize=8)
    ax.set_title("Plantilla más cercana a cada tramo (verde: la de su propia fase)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "figuras" / "plantillas_modelo_vs_series.png", dpi=150)
    plt.close(fig)

    print({f: len(T) for f, T in todas.plantillas.items()})
    print(tabla[["serie_id", "fase", "dias", "largo_plantilla", "plantilla_mas_cercana", "acierto"]]
          .to_string(index=False))
    print(f"\nAciertos: {tabla.acierto.sum()} de {len(tabla)}")
    print(tabla.groupby("fase", sort=False).acierto.sum())


if __name__ == "__main__":
    main()

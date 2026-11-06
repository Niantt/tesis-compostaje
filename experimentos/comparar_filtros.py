"""
comparar_filtros.py — Tesis INFO-1197, Etapa 3
Compara los filtros de ruido (retroalimentación de la Etapa 2):

  - Savitzky-Golay (ventana 7, orden 2)                  -> línea base de la Etapa 2
  - Mediana móvil + media móvil, centradas (varias ventanas)
  - Mediana móvil + media móvil, causales (solo días pasados)
  - Polinomio de grado 12 por curva

Mediciones sobre las 33 series:
  1. Fidelidad: RMSE entre la curva filtrada y las mediciones.
  2. Ruido residual: desviación de la segunda diferencia de la curva (°C/día²);
     menor = curva más suave.
  3. Desfase del pico: días entre el máximo de la curva filtrada y el máximo medido.
  4. Efecto en las técnicas: exactitud y F1 macro (LOO) de puntos de cambio, DTW e
     híbrido, y coherencia en las 33 series.

Uso: python experimentos/comparar_filtros.py [ruta.xlsx] [--salida salida_etapa3]
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compostaje import Preprocesador  # noqa: E402
from compostaje.compilacion import cargar_conjunto  # noqa: E402
from evaluar_tecnicas import coherencia, evaluar_loo, resumen_loo, tecnicas  # noqa: E402

warnings.filterwarnings("ignore")

FILTROS = {
    "Savitzky-Golay (7, 2)": dict(filtro="savgol"),
    "Mediana 3 + media 3": dict(filtro="mediana_media", ventana_mediana=3, ventana_media=3),
    "Mediana 5 + media 3": dict(filtro="mediana_media", ventana_mediana=5, ventana_media=3),
    "Mediana 5 + media 5": dict(filtro="mediana_media", ventana_mediana=5, ventana_media=5),
    "Mediana 7 + media 3": dict(filtro="mediana_media", ventana_mediana=7, ventana_media=3),
    "Mediana 5 + media 3 (causal)": dict(filtro="mediana_media", ventana_mediana=5,
                                          ventana_media=3, causal=True),
    "Polinomio grado 12": dict(filtro="polinomio", grado=12),
}
EJEMPLOS = ["PILA_15", "DOCTORAL_PILA_1", "PENAILILLO_C1_SH", "PILA_23"]


def calidad(series):
    filas = []
    for s in series:
        med = s.metadatos["es_medicion"]
        y_med = s.metadatos["temp_interp"][med]
        y_f = s.temperaturas
        filas.append(dict(rmse=np.sqrt(np.mean((y_f[med] - y_med) ** 2)),
                          ruido=np.std(np.diff(y_f, 2)) if len(y_f) > 2 else np.nan,
                          desfase_pico=abs(int(np.argmax(y_f)) - int(s.tiempos[med][np.argmax(y_med)])),
                          grado=s.metadatos.get("grado_polinomio", np.nan)))
    return pd.DataFrame(filas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx", nargs="?", default="Compilacion_temperatura_compost.xlsx")
    ap.add_argument("--salida", default="salida_etapa3")
    args = ap.parse_args()
    out = Path(args.salida)
    (out / "figuras").mkdir(parents=True, exist_ok=True)

    filas, curvas = [], {}
    for nombre, cfg in FILTROS.items():
        series, _, _ = cargar_conjunto(args.xlsx, Preprocesador(**cfg))
        curvas[nombre] = {s.nombre: s for s in series}
        c = calidad(series)
        etiquetadas = [s for s in series if s.etiquetada]
        pred = evaluar_loo(etiquetadas)
        res = resumen_loo(etiquetadas, pred, list(tecnicas(None)))
        # coherencia en las 33 series (modelo con todas las etiquetadas; LOO para ellas)
        from compostaje import ModeloFases
        modelo = ModeloFases(21).entrenar(etiquetadas)
        resultados = {n: {s.nombre: pred.get((n, s.nombre)) or t.identificar(s) for s in series}
                      for n, t in tecnicas(modelo).items()}
        coh = coherencia(resultados, series).set_index("tecnica")
        fila = dict(filtro=nombre, rmse_c=c.rmse.mean(), ruido=c.ruido.mean(),
                    desfase_pico_dias=c.desfase_pico.mean(),
                    series_grado_reducido=int((c.grado < 12).sum()) if c.grado.notna().any() else 0)
        for _, r in res.iterrows():
            corto = {"Puntos de cambio": "cp", "DTW": "dtw", "Híbrido sin restricción": "hib_sr",
                     "Híbrido con orden": "hib_o", "Híbrido con orden y regla 40 °C": "hib",
                     "Híbrido, cambios por derivada": "hib_der"}[r.tecnica]
            fila[f"{corto}_exactitud"] = r.exactitud
            fila[f"{corto}_f1"] = r.f1_macro
            fila[f"{corto}_error_dias"] = r.error_medio_dias
            fila[f"{corto}_coherentes"] = coh.loc[r.tecnica, "coherentes"]
        filas.append(fila)
        print(f"  {nombre:<30} RMSE={fila['rmse_c']:.2f}  ruido={fila['ruido']:.2f}  "
              f"CP F1={fila['cp_f1']:.3f}  DTW F1={fila['dtw_f1']:.3f}  Híbrido F1={fila['hib_f1']:.3f} "
              f"(error {fila['hib_error_dias']:.1f} d)")
    tabla = pd.DataFrame(filas).round(3)
    tabla.to_csv(out / "comparacion_filtros.csv", index=False)

    # figura: ejemplos con los filtros principales
    principales = ["Savitzky-Golay (7, 2)", "Mediana 5 + media 3", "Mediana 5 + media 3 (causal)",
                   "Polinomio grado 12"]
    colores = ["k", "#de2d26", "#3182bd", "#31a354"]
    fig, axs = plt.subplots(1, len(EJEMPLOS), figsize=(5 * len(EJEMPLOS), 3.8), sharey=True)
    for ax, sid in zip(axs, EJEMPLOS):
        s0 = curvas[principales[0]][sid]
        med = s0.metadatos["es_medicion"]
        ax.plot(s0.tiempos[med], s0.metadatos["temp_interp"][med], ".", color="0.55", ms=5,
                label="Medición")
        for nombre, c in zip(principales, colores):
            s = curvas[nombre][sid]
            ax.plot(s.tiempos, s.temperaturas, color=c, lw=1.4, label=nombre)
        ax.set_title(sid)
        ax.set_xlabel("Días")
    axs[0].set_ylabel("Temperatura (°C)")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "figuras" / "comparacion_filtros.png", dpi=150)
    plt.close(fig)
    pd.set_option("display.width", 250)
    print("\n", tabla[["filtro", "rmse_c", "ruido", "desfase_pico_dias", "cp_f1", "dtw_f1", "hib_f1",
                       "hib_error_dias", "dtw_coherentes", "hib_coherentes"]].to_string(index=False))


if __name__ == "__main__":
    main()

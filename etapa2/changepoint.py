"""
changepoint.py — Tesis INFO-1197
Identificación de fases del compostaje mediante detección de puntos de cambio.

Idea:
  El proceso tiene 4 fases (mesófila, termófila, enfriamiento, maduración), por lo
  que se buscan exactamente 3 puntos de cambio en cada serie. Los segmentos
  resultantes se asignan a las fases en orden.

Pasos:
  1. Evalúa varias combinaciones (función de costo x algoritmo) sobre las series
     con fases etiquetadas: error en días de cada límite, exactitud y F1 por fase.
  2. Elige la mejor combinación (menor error medio en días).
  3. La aplica a todas las series seleccionadas y genera figuras.

Entrada : salida/datos_diarios_preprocesados.csv y salida/etiquetas_fase.csv
          (generados por preprocesamiento.py)
Uso     : python changepoint.py  [--entrada salida] [--salida salida_changepoint]
Requiere: pip install ruptures
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.metrics import accuracy_score, f1_score

FASES = ["Mesófila", "Termófila", "Enfriamiento", "Maduración"]
N_CAMBIOS = len(FASES) - 1      # 3 puntos de cambio -> 4 segmentos
MIN_DIAS = 2                    # largo mínimo de un segmento (días)
COLORES = {"Mesófila": "#9ecae1", "Termófila": "#fc9272",
           "Enfriamiento": "#fdd0a2", "Maduración": "#a1d99b"}

# Combinaciones a comparar
#   l2      : cambios en el nivel medio de temperatura
#   clinear : tramos lineales continuos (cambios de pendiente sin saltos)
#   linear  : tramos lineales independientes (regresión por segmento)
COSTOS = ["l2", "clinear", "linear"]
ALGORITMOS = {"Dynp": rpt.Dynp,       # programación dinámica: óptimo exacto
              "Binseg": rpt.Binseg}   # segmentación binaria: aproximado y rápido


# ---------------------------------------------------------------------------
def preparar_senal(y, costo):
    """Formato de entrada que espera cada función de costo de ruptures."""
    if costo == "linear":
        # columna 0 = variable respuesta; resto = regresores (intercepto y tiempo)
        return np.column_stack([y, np.ones_like(y), np.arange(len(y))])
    return y.reshape(-1, 1)


def detectar(y, costo, algoritmo):
    """Devuelve los 3 puntos de cambio (día en que inicia cada fase 2, 3 y 4)."""
    clase = ALGORITMOS[algoritmo]
    # jump=1: se evalúan todos los días como posibles cambios (por defecto ruptures usa 5)
    modelo = clase(model=costo, min_size=MIN_DIAS, jump=1).fit(preparar_senal(y, costo))
    return modelo.predict(n_bkps=N_CAMBIOS)[:-1]   # el último es el fin de la serie


def fases_por_dia(n_dias, cambios):
    """Asigna a cada día la fase según el segmento en que cae."""
    idx = np.searchsorted(np.asarray(cambios), np.arange(n_dias), side="right")
    return np.array(FASES)[idx]


def limites_reales(etiquetas, sid):
    """Días (base 0) en que comienzan termófila, enfriamiento y maduración."""
    e = etiquetas[etiquetas.serie_id == sid].sort_values("dia_inicio")
    return [int(d) - 1 for d in e["dia_inicio"]][1:]


# ---------------------------------------------------------------------------
def evaluar(diario, etiquetas):
    filas = []
    ids = etiquetas["serie_id"].unique()
    for costo in COSTOS:
        for alg in ALGORITMOS:
            errores, y_real, y_pred = [], [], []
            for sid in ids:
                d = diario[diario.serie_id == sid]
                y = d["temp_suav_c"].to_numpy()
                pred = detectar(y, costo, alg)
                real = limites_reales(etiquetas, sid)
                errores.append(np.abs(np.array(pred) - np.array(real)))
                mask = d["fase_ref"].notna().to_numpy()
                y_real += list(d["fase_ref"].to_numpy()[mask])
                y_pred += list(fases_por_dia(len(y), pred)[mask])
            err = np.mean(errores, axis=0)
            filas.append(dict(
                costo=costo, algoritmo=alg,
                error_termofila=err[0], error_enfriamiento=err[1], error_maduracion=err[2],
                error_medio_dias=err.mean(),
                exactitud=accuracy_score(y_real, y_pred),
                f1_macro=f1_score(y_real, y_pred, labels=FASES, average="macro", zero_division=0),
                **{f"f1_{f.lower()}": v for f, v in zip(
                    FASES, f1_score(y_real, y_pred, labels=FASES, average=None, zero_division=0))}))
    return pd.DataFrame(filas).sort_values("error_medio_dias").round(3)


def tabla_latex(ev):
    lineas = [r"\begin{table}[htbp]", r"\centering", r"\small",
              r"\caption{Evaluación de la detección de puntos de cambio sobre las series "
              r"con fases etiquetadas. Error: diferencia absoluta media, en días, entre el "
              r"inicio detectado y el inicio de referencia de cada fase.}",
              r"\label{tab:eval_changepoint}",
              r"\resizebox{\textwidth}{!}{%",
              r"\begin{tabular}{llccccc}", r"\hline",
              r" & & \multicolumn{3}{c}{Error en el inicio de (días)} & & \\",
              r"Costo & Algoritmo & Termófila & Enfriam. & Maduración & Exactitud & F1 macro \\",
              r"\hline"]
    coma = lambda v, n: f"{v:.{n}f}".replace(".", ",")   # coma decimal
    for _, r in ev.iterrows():
        lineas.append(f"{r.costo} & {r.algoritmo} & {coma(r.error_termofila, 1)} & "
                      f"{coma(r.error_enfriamiento, 1)} & {coma(r.error_maduracion, 1)} & "
                      f"{coma(r.exactitud, 2)} & {coma(r.f1_macro, 2)} \\\\")
    lineas += [r"\hline", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lineas)


# ---------------------------------------------------------------------------
def sombrear(ax, cambios, n):
    bordes = [0] + list(cambios) + [n]
    for fase, a, b in zip(FASES, bordes[:-1], bordes[1:]):
        ax.axvspan(a, b, color=COLORES[fase], alpha=0.6, label=fase)


def fig_etiquetadas(diario, etiquetas, deteccion, ruta):
    ids = list(etiquetas["serie_id"].unique())
    fig, axs = plt.subplots(2, 3, figsize=(16, 7), sharey=True)
    for ax, sid in zip(axs.flat, ids):
        d = diario[diario.serie_id == sid]
        cambios = deteccion.loc[sid, ["cambio_1", "cambio_2", "cambio_3"]].tolist()
        sombrear(ax, cambios, len(d))
        ax.plot(d.t_dias, d.temp_suav_c, color="k", lw=1.2)
        for i, x in enumerate(limites_reales(etiquetas, sid)):
            ax.axvline(x, color="navy", ls="--", lw=1.2, label="referencia" if i == 0 else None)
        ax.set_title(sid)
    for ax in axs.flat[len(ids):]:
        ax.axis("off")
    axs.flat[0].legend(fontsize=8, loc="upper right")
    fig.supxlabel("Días")
    fig.supylabel("Temperatura suavizada (°C)")
    fig.suptitle("Fases detectadas (color) frente a límites de referencia (línea discontinua)")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def fig_todas(diario, deteccion, ruta):
    ids = list(deteccion.index)
    ncol = 6
    nfil = int(np.ceil(len(ids) / ncol))
    fig, axs = plt.subplots(nfil, ncol, figsize=(18, 2.3 * nfil), sharey=True)
    for ax, sid in zip(axs.flat, ids):
        d = diario[diario.serie_id == sid]
        sombrear(ax, deteccion.loc[sid, ["cambio_1", "cambio_2", "cambio_3"]].tolist(), len(d))
        ax.plot(d.t_dias, d.temp_suav_c, color="k", lw=1)
        ax.set_title(sid, fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axs.flat[len(ids):]:
        ax.axis("off")
    handles, labels = axs.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", ncol=4)
    fig.supxlabel("Días")
    fig.supylabel("Temperatura (°C)")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", default="salida")
    ap.add_argument("--salida", default="salida_changepoint")
    args = ap.parse_args()
    ent, out = Path(args.entrada), Path(args.salida)
    out.mkdir(parents=True, exist_ok=True)

    diario = pd.read_csv(ent / "datos_diarios_preprocesados.csv")
    etiquetas = pd.read_csv(ent / "etiquetas_fase.csv")

    # 1-2. evaluación y elección de la mejor combinación
    ev = evaluar(diario, etiquetas)
    ev.to_csv(out / "evaluacion_changepoint.csv", index=False)
    (out / "tabla_eval_changepoint.tex").write_text(tabla_latex(ev), encoding="utf-8")
    mejor = ev.iloc[0]
    print("Evaluación sobre series etiquetadas:\n",
          ev[["costo", "algoritmo", "error_termofila", "error_enfriamiento",
              "error_maduracion", "error_medio_dias", "exactitud", "f1_macro"]].to_string(index=False))
    print(f"\nMejor combinación: costo={mejor.costo}, algoritmo={mejor.algoritmo}")

    # 3. aplicar a todas las series
    filas, por_dia = [], []
    for sid, d in diario.groupby("serie_id", sort=False):
        y = d["temp_suav_c"].to_numpy()
        c = detectar(y, mejor.costo, mejor.algoritmo)
        filas.append(dict(serie_id=sid, cambio_1=c[0], cambio_2=c[1], cambio_3=c[2],
                          duracion_dias=len(y)))
        por_dia.append(d[["serie_id", "t_dias", "temp_suav_c", "fase_ref"]]
                       .assign(fase_detectada=fases_por_dia(len(y), c)))
    deteccion = pd.DataFrame(filas).set_index("serie_id")
    deteccion.to_csv(out / "puntos_de_cambio.csv")
    pd.concat(por_dia).to_csv(out / "fases_detectadas_por_dia.csv", index=False)

    fig_etiquetadas(diario, etiquetas, deteccion, out / "changepoint_etiquetadas.png")
    fig_todas(diario, deteccion, out / "changepoint_todas.png")
    print(f"\nSeries procesadas: {len(deteccion)}. Resultados en '{out}/'.")


if __name__ == "__main__":
    main()

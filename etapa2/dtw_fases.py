"""
dtw_fases.py — Tesis INFO-1197
Identificación de fases del compostaje mediante Dynamic Time Warping (DTW).

Idea:
  Cada día de una serie se representa por la ventana de W días centrada en él.
  Esa ventana se compara con DTW contra ventanas de referencia de cada fase y se
  le asigna la fase más parecida. A diferencia de los puntos de cambio, las fases
  NO se asignan por orden, sino por la forma y el nivel de la curva.

Dos variantes de "modelo por fase":
  - Plantilla DBA : una curva típica por fase (promedio DBA de sus ventanas).
                    Se asigna la fase de la plantilla más cercana.
  - 1-NN          : todas las ventanas etiquetadas de cada fase son el modelo.
                    Se asigna la fase de la ventana más cercana (vecino más próximo).

Evaluación:
  Validación dejando una serie fuera (leave-one-out): para evaluar una serie
  etiquetada, los modelos se construyen solo con las demás.

Entrada : salida/datos_diarios_preprocesados.csv  (preprocesamiento.py)
          salida_changepoint/fases_detectadas_por_dia.csv (opcional, para comparar)
Uso     : python dtw_fases.py  [--entrada salida] [--salida salida_dtw]
Requiere: pip install tslearn
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.metrics import cdist_dtw

warnings.filterwarnings("ignore")

FASES = ["Mesófila", "Termófila", "Enfriamiento", "Maduración"]
COLORES = {"Mesófila": "#9ecae1", "Termófila": "#fc9272",
           "Enfriamiento": "#fdd0a2", "Maduración": "#a1d99b"}

# Configuraciones a comparar
VENTANAS = [7, 11, 15, 21]                   # días por ventana
SENALES = {"temp_suav_c": "°C",              # temperatura suavizada
           "temp_norm": "min-max"}           # temperatura normalizada por serie
MODELOS = ["Plantilla DBA", "1-NN"]
RADIO = 2                                    # banda de Sakoe-Chiba (días de desfase permitido)
DTW_PARAMS = dict(global_constraint="sakoe_chiba", sakoe_chiba_radius=RADIO)


# ---------------------------------------------------------------------------
def ventanas(y, w):
    """Ventana de w días centrada en cada día (los bordes se rellenan repitiendo el extremo)."""
    h = w // 2
    yp = np.pad(y, h, mode="edge")
    return np.stack([yp[i:i + w] for i in range(len(y))])[..., None]


def datos_entrenamiento(diario, ids, senal, w):
    X, y = [], []
    for sid in ids:
        g = diario[diario.serie_id == sid]
        X.append(ventanas(g[senal].to_numpy(), w))
        y += list(g["fase_ref"])
    return np.vstack(X), np.array(y)


def construir_plantillas(X, y):
    """Modelo por fase: una curva promedio DBA para cada fase."""
    return {f: dtw_barycenter_averaging(X[y == f], max_iter=10, metric_params=DTW_PARAMS)
            for f in FASES}


def clasificar(X, Xtr, ytr, modelo, plantillas=None):
    if modelo == "Plantilla DBA":
        T = np.stack([plantillas[f] for f in FASES])
        D = cdist_dtw(X, T, **DTW_PARAMS)
        return np.array(FASES)[D.argmin(axis=1)]
    D = cdist_dtw(X, Xtr, **DTW_PARAMS)
    return ytr[D.argmin(axis=1)]


# ---------------------------------------------------------------------------
def evaluar_config(diario, etiquetadas, senal, w, modelo):
    """Leave-one-out sobre las series etiquetadas. Devuelve métricas y predicciones."""
    y_real, y_pred, pred_por_serie = [], [], {}
    for sid in etiquetadas:
        resto = [s for s in etiquetadas if s != sid]
        Xtr, ytr = datos_entrenamiento(diario, resto, senal, w)
        plantillas = construir_plantillas(Xtr, ytr) if modelo == "Plantilla DBA" else None
        g = diario[diario.serie_id == sid]
        p = clasificar(ventanas(g[senal].to_numpy(), w), Xtr, ytr, modelo, plantillas)
        pred_por_serie[sid] = p
        y_real += list(g["fase_ref"])
        y_pred += list(p)
    f1s = f1_score(y_real, y_pred, labels=FASES, average=None, zero_division=0)
    metricas = dict(senal=SENALES[senal], ventana=w, modelo=modelo,
                    exactitud=accuracy_score(y_real, y_pred),
                    f1_macro=f1s.mean(),
                    **{f"f1_{f.lower()}": v for f, v in zip(FASES, f1s)})
    return metricas, pred_por_serie


def tabla_latex(ev, etiqueta, titulo):
    coma = lambda v: f"{v:.2f}".replace(".", ",")
    lineas = [r"\begin{table}[htbp]", r"\centering", r"\small",
              r"\caption{" + titulo + "}",
              r"\label{" + etiqueta + "}",
              r"\resizebox{\textwidth}{!}{%",
              r"\begin{tabular}{lcl" + "c" * 6 + "}", r"\hline",
              r" & & & & & \multicolumn{4}{c}{F1 por fase} \\",
              r"Señal & $W$ & Modelo & Exactitud & F1 macro & Mes. & Term. & Enfr. & Mad. \\",
              r"\hline"]
    for _, r in ev.iterrows():
        lineas.append(f"{r.senal} & {r.ventana} & {r.modelo} & {coma(r.exactitud)} & "
                      f"{coma(r.f1_macro)} & {coma(r['f1_mesófila'])} & {coma(r['f1_termófila'])} & "
                      f"{coma(r.f1_enfriamiento)} & {coma(r['f1_maduración'])} \\\\")
    lineas += [r"\hline", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lineas).replace("°C", r"\textdegree{}C")


# ---------------------------------------------------------------------------
def sombrear_por_dia(ax, fases):
    """Pinta el fondo según la fase asignada a cada día (agrupa días consecutivos)."""
    inicio = 0
    for i in range(1, len(fases) + 1):
        if i == len(fases) or fases[i] != fases[inicio]:
            ax.axvspan(inicio, i, color=COLORES[fases[inicio]], alpha=0.6, lw=0)
            inicio = i


def leyenda(fig):
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORES[f], alpha=0.6) for f in FASES]
    fig.legend(handles, FASES, loc="lower right", ncol=4)


def fig_plantillas(plantillas, senal, w, ruta):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(w) - w // 2
    lineas = {"Mesófila": "#3182bd", "Termófila": "#de2d26",
              "Enfriamiento": "#e6550d", "Maduración": "#31a354"}
    for f in FASES:
        ax.plot(x, plantillas[f].ravel(), lw=2.2, label=f, color=lineas[f])
    ax.set_xlabel("Días relativos al centro de la ventana")
    ax.set_ylabel(f"Temperatura ({SENALES[senal]})")
    ax.set_title(f"Plantilla DBA de cada fase (ventana de {w} días)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def fig_etiquetadas(diario, pred, ruta):
    ids = list(pred)
    fig, axs = plt.subplots(len(ids), 1, figsize=(13, 2.1 * len(ids)))
    for ax, sid in zip(np.atleast_1d(axs), ids):
        g = diario[diario.serie_id == sid]
        n = len(g)
        sombrear_por_dia(ax, pred[sid])                     # fondo: predicción DTW
        ax.plot(g.t_dias, g.temp_suav_c, color="k", lw=1.2)
        ref = g["fase_ref"].to_numpy()
        for i in range(n):                                   # franja superior: referencia
            ax.add_patch(plt.Rectangle((i, 66), 1, 6, color=COLORES[ref[i]], lw=0))
        ax.set_ylim(0, 72)
        ax.set_xlim(0, n)
        ax.set_title(f"{sid}  (franja superior: referencia · fondo: DTW)", fontsize=9)
    leyenda(fig)
    fig.supxlabel("Días")
    fig.supylabel("Temperatura suavizada (°C)")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def fig_todas(diario, pred, ruta):
    ids = list(pred)
    ncol = 6
    nfil = int(np.ceil(len(ids) / ncol))
    fig, axs = plt.subplots(nfil, ncol, figsize=(18, 2.3 * nfil), sharey=True)
    for ax, sid in zip(axs.flat, ids):
        g = diario[diario.serie_id == sid]
        sombrear_por_dia(ax, pred[sid])
        ax.plot(g.t_dias, g.temp_suav_c, color="k", lw=1)
        ax.set_xlim(0, len(g))
        ax.set_title(sid, fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axs.flat[len(ids):]:
        ax.axis("off")
    leyenda(fig)
    fig.supxlabel("Días")
    fig.supylabel("Temperatura (°C)")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def comparar_con_changepoint(diario_lab, pred_dtw, ruta_cp, out):
    """Métricas de ambas técnicas sobre los mismos días etiquetados."""
    if not ruta_cp.exists():
        print("(No se encontró la salida de changepoint; se omite la comparación.)")
        return None
    cp = pd.read_csv(ruta_cp)
    filas = []
    for nombre, pred in [("Puntos de cambio", None), ("DTW", pred_dtw)]:
        y_real, y_pred = [], []
        for sid, g in diario_lab.groupby("serie_id", sort=False):
            y_real += list(g["fase_ref"])
            if pred is None:
                y_pred += list(cp[cp.serie_id == sid].sort_values("t_dias")["fase_detectada"])
            else:
                y_pred += list(pred[sid])
        f1s = f1_score(y_real, y_pred, labels=FASES, average=None, zero_division=0)
        filas.append(dict(tecnica=nombre, exactitud=accuracy_score(y_real, y_pred),
                          f1_macro=f1s.mean(), **{f"f1_{f.lower()}": v for f, v in zip(FASES, f1s)}))
    comp = pd.DataFrame(filas).round(3)
    comp.to_csv(out / "comparacion_tecnicas.csv", index=False)
    return comp


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", default="salida")
    ap.add_argument("--changepoint", default="salida_changepoint")
    ap.add_argument("--salida", default="salida_dtw")
    args = ap.parse_args()
    ent, out = Path(args.entrada), Path(args.salida)
    out.mkdir(parents=True, exist_ok=True)

    diario = pd.read_csv(ent / "datos_diarios_preprocesados.csv")
    etiquetadas = list(diario.loc[diario.fase_ref.notna(), "serie_id"].unique())
    print(f"Series etiquetadas: {len(etiquetadas)} | Configuraciones a evaluar: "
          f"{len(SENALES) * len(VENTANAS) * len(MODELOS)}")

    # 1. evaluación leave-one-out de todas las configuraciones
    resultados, predicciones = [], {}
    for senal in SENALES:
        for w in VENTANAS:
            for modelo in MODELOS:
                m, p = evaluar_config(diario, etiquetadas, senal, w, modelo)
                resultados.append(m)
                predicciones[(senal, w, modelo)] = p
                print(f"  {SENALES[senal]:>7} W={w:<2} {modelo:<13} "
                      f"exactitud={m['exactitud']:.3f}  F1 macro={m['f1_macro']:.3f}")
    ev = pd.DataFrame(resultados).sort_values("f1_macro", ascending=False).round(3)
    ev.to_csv(out / "evaluacion_dtw.csv", index=False)
    # tabla completa (anexo) y resumen con la mejor ventana por señal y modelo (cuerpo)
    (out / "tabla_eval_dtw_completa.tex").write_text(tabla_latex(
        ev, "tab:eval_dtw_completa",
        "Evaluación de las 16 configuraciones de DTW sobre las series con fases etiquetadas "
        "(validación dejando una serie fuera), ordenadas por F1 macro."), encoding="utf-8")
    resumen = ev.groupby(["senal", "modelo"], sort=False).head(1)
    (out / "tabla_eval_dtw_resumen.tex").write_text(tabla_latex(
        resumen, "tab:eval_dtw",
        "Mejor configuración de DTW para cada combinación de señal y modelo "
        "(validación dejando una serie fuera)."), encoding="utf-8")

    mejor = ev.iloc[0]
    senal = next(k for k, v in SENALES.items() if v == mejor.senal)
    w, modelo = int(mejor.ventana), mejor.modelo
    print(f"\nMejor configuración: señal={mejor.senal}, ventana={w} días, modelo={modelo} "
          f"(exactitud={mejor.exactitud:.3f}, F1 macro={mejor.f1_macro:.3f})")

    # 2. modelo final entrenado con todas las series etiquetadas
    Xtr, ytr = datos_entrenamiento(diario, etiquetadas, senal, w)
    plantillas = construir_plantillas(Xtr, ytr)
    fig_plantillas(plantillas, senal, w, out / "dtw_plantillas_por_fase.png")

    pred_lab = predicciones[(senal, w, modelo)]         # predicciones leave-one-out
    pred, filas = {}, []
    for sid, g in diario.groupby("serie_id", sort=False):
        if sid in pred_lab:
            p = pred_lab[sid]
        else:
            p = clasificar(ventanas(g[senal].to_numpy(), w), Xtr, ytr, modelo, plantillas)
        pred[sid] = p
        filas.append(g[["serie_id", "t_dias", "temp_suav_c", "fase_ref"]].assign(fase_dtw=p))
    pd.concat(filas).to_csv(out / "fases_dtw_por_dia.csv", index=False)

    fig_etiquetadas(diario, pred_lab, out / "dtw_etiquetadas.png")
    fig_todas(diario, pred, out / "dtw_todas.png")

    # 3. comparación con puntos de cambio
    comp = comparar_con_changepoint(diario[diario.serie_id.isin(etiquetadas)], pred_lab,
                                    Path(args.changepoint) / "fases_detectadas_por_dia.csv", out)
    if comp is not None:
        print("\nComparación sobre las series etiquetadas:\n", comp.to_string(index=False))
    print(f"\nSeries procesadas: {len(pred)}. Resultados en '{out}/'.")


if __name__ == "__main__":
    main()

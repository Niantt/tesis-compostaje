"""
evaluar_tecnicas.py — Tesis INFO-1197, Etapa 3
Compara las técnicas de identificación de fases antes y después de la mejora:

  Antes (Etapa 2):  Puntos de cambio · DTW
  Después (Etapa 3): Híbrido sin restricción · Híbrido con restricción de orden

Mediciones:
  1. Series etiquetadas, validación dejando una serie fuera (LOO): exactitud,
     F1 macro, F1 por fase y error en días del inicio de cada fase.
  2. Las 33 series del conjunto: coherencia de la secuencia de fases (porcentaje de
     series con transiciones válidas), series que llegan a maduración y retrocesos.
  3. Sensibilidad del híbrido al número de tramos.
  4. Tiempo de ejecución por serie.

Uso: python experimentos/evaluar_tecnicas.py [ruta.xlsx] [--salida salida_etapa3]
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compostaje import (COLORES, FASES, TRANSICIONES_VALIDAS, IdentificadorDTW,  # noqa: E402
                        IdentificadorHibrido, IdentificadorPuntosCambio, ModeloFases)
from compostaje.compilacion import cargar_conjunto  # noqa: E402

VENTANA, RADIO = 21, 2           # configuración de DTW elegida en la Etapa 2 (D-08)
N_TRAMOS = [4, 6, 8, 10, 12, 20, 30, 40]   # número fijo de tramos evaluado para el híbrido
UMBRALES_DERIVADA = [0.5, 1.0, 1.5, 2.0, 3.0]  # °C/día para contar episodios de cambio
N_TRAMOS_DEFECTO = 8
UMBRAL_DEFECTO = 1.0
FINAL = "Híbrido, cambios por derivada"


def tecnicas(modelo, n_tramos=N_TRAMOS_DEFECTO):
    return {
        "Puntos de cambio": IdentificadorPuntosCambio(),
        "DTW": IdentificadorDTW(modelo, RADIO),
        "Híbrido sin restricción": IdentificadorHibrido(modelo, RADIO, n_tramos, restringir=False),
        "Híbrido con orden": IdentificadorHibrido(modelo, RADIO, n_tramos, restringir=True,
                                                  umbral_termofila=None),
        "Híbrido con orden y regla 40 °C": IdentificadorHibrido(modelo, RADIO, n_tramos),
        "Híbrido, cambios por derivada": IdentificadorHibrido(modelo, RADIO, "derivada",
                                                              umbral_derivada=UMBRAL_DEFECTO),
    }


# ---------------------------------------------------------------------------
def inicios(fases):
    """Primer día de Termófila, Enfriamiento y Maduración (NaN si no aparece)."""
    return [float(np.argmax(fases == f)) if np.any(fases == f) else np.nan for f in FASES[1:]]


def metricas(y_real, y_pred):
    f1s = f1_score(y_real, y_pred, labels=FASES, average=None, zero_division=0)
    return dict(exactitud=accuracy_score(y_real, y_pred), f1_macro=f1s.mean(),
                **{f"f1_{f.lower()}": v for f, v in zip(FASES, f1s)})


def evaluar_loo(etiquetadas, n_tramos=N_TRAMOS_DEFECTO):
    """Predicciones de cada técnica en cada serie etiquetada, entrenando DTW sin ella."""
    pred = {}
    for s in etiquetadas:
        modelo = ModeloFases(VENTANA).entrenar([o for o in etiquetadas if o is not s])
        for nombre, tec in tecnicas(modelo, n_tramos).items():
            pred[(nombre, s.nombre)] = tec.identificar(s)
    return pred


def evaluar_loo_una(etiquetadas, fabrica, nombre="x"):
    """LOO de una sola técnica: fabrica(modelo) -> identificador."""
    pred = {}
    for s in etiquetadas:
        modelo = ModeloFases(VENTANA).entrenar([o for o in etiquetadas if o is not s])
        pred[(nombre, s.nombre)] = fabrica(modelo).identificar(s)
    return resumen_loo(etiquetadas, pred, [nombre]).iloc[0]


def resumen_loo(etiquetadas, pred, nombres):
    filas = []
    for nombre in nombres:
        y_real, y_pred, errores = [], [], []
        for s in etiquetadas:
            m = np.array([f is not None for f in s.fase_ref])
            r = pred[(nombre, s.nombre)]
            y_real += list(s.fase_ref[m])
            y_pred += list(r.fase_por_dia[m])
            errores.append(np.abs(np.array(inicios(r.fase_por_dia)) - np.array(inicios(s.fase_ref))))
        err = np.nanmean(errores, axis=0)
        filas.append(dict(tecnica=nombre, **metricas(y_real, y_pred),
                          error_termofila=err[0], error_enfriamiento=err[1],
                          error_maduracion=err[2], error_medio_dias=np.nanmean(err)))
    return pd.DataFrame(filas)


def coherencia(resultados, series):
    """Indicadores de coherencia sobre todas las series (no requieren etiquetas)."""
    filas = []
    for nombre, por_serie in resultados.items():
        coh = [r.es_coherente() for r in por_serie.values()]
        mad = [np.any(r.fase_por_dia == "Maduración") for r in por_serie.values()]
        inicio_mes = [r.fase_por_dia[0] == "Mesófila" for r in por_serie.values()]
        invalidas, term_frias = [], 0
        por_id = {x.nombre: x for x in series}
        for sid, r in por_serie.items():
            y = por_id[sid].metadatos["temp_interp"]
            term_frias += sum(1 for f, a, b in r.segmentos()
                              if f == "Termófila" and y[int(a):int(b)].max() < 40)
            orden = [t[0] for t in r.segmentos()]
            invalidas.append(sum((a, b) not in TRANSICIONES_VALIDAS for a, b in zip(orden, orden[1:])))
        filas.append(dict(tecnica=nombre, series=len(coh),
                          coherentes=int(np.sum(coh)), pct_coherentes=100 * np.mean(coh),
                          inician_mesofila=int(np.sum(inicio_mes)),
                          llegan_maduracion=int(np.sum(mad)),
                          transiciones_invalidas=int(np.sum(invalidas)),
                          termofila_bajo_40=term_frias))
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
def sombrear(ax, fases):
    ini = 0
    for i in range(1, len(fases) + 1):
        if i == len(fases) or fases[i] != fases[ini]:
            ax.axvspan(ini, i, color=COLORES[fases[ini]], alpha=0.6, lw=0)
            ini = i


def fig_antes_despues(series, resultados, ids, ruta, tecnicas_fig):
    ids = [i for i in ids if i in {s.nombre for s in series}]
    por_id = {s.nombre: s for s in series}
    fig, axs = plt.subplots(len(ids), len(tecnicas_fig), figsize=(5 * len(tecnicas_fig), 2.4 * len(ids)),
                            squeeze=False, sharey=True)
    for i, sid in enumerate(ids):
        s = por_id[sid]
        for j, nombre in enumerate(tecnicas_fig):
            ax = axs[i, j]
            sombrear(ax, resultados[nombre][sid].fase_por_dia)
            ax.plot(s.tiempos, s.temperaturas, color="k", lw=1)
            ax.set_xlim(0, len(s))
            if i == 0:
                ax.set_title(nombre, fontsize=10)
            if j == 0:
                ax.set_ylabel(sid, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORES[f], alpha=0.6) for f in FASES]
    fig.legend(handles, FASES, loc="lower center", ncol=4)
    fig.supxlabel("Días", y=0.04)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def fig_todas(series, resultados, nombre, ruta):
    ncol = 6
    nfil = int(np.ceil(len(series) / ncol))
    fig, axs = plt.subplots(nfil, ncol, figsize=(18, 2.3 * nfil), sharey=True)
    for ax, s in zip(axs.flat, series):
        sombrear(ax, resultados[nombre][s.nombre].fase_por_dia)
        ax.plot(s.tiempos, s.temperaturas, color="k", lw=1)
        ax.set_xlim(0, len(s))
        ax.set_title(s.nombre, fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axs.flat[len(series):]:
        ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORES[f], alpha=0.6) for f in FASES]
    fig.legend(handles, FASES, loc="lower right", ncol=4)
    fig.supxlabel("Días")
    fig.supylabel("Temperatura (°C)")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def fig_sensibilidad(sens, ruta):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    fijo, der = sens[sens.criterio == "fijo"], sens[sens.criterio == "derivada"]
    for ax, col, titulo in [(axs[0], "f1_macro", "F1 macro"), (axs[1], "error_medio_dias", "Error en límites (días)")]:
        ax.plot(fijo.tramos_medios, fijo[col], "o-", color="#888888", label="Número fijo de tramos")
        ax.plot(der.tramos_medios, der[col], "s-", color="#de2d26", label="Derivada discreta (promedio)")
        for _, r in der.iterrows():
            ax.annotate(f"{r.parametro:g} °C/d", (r.tramos_medios, r[col]), fontsize=7,
                        textcoords="offset points", xytext=(4, 4))
        ax.set_xlabel("Tramos por serie")
        ax.set_title(titulo, fontsize=10)
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def tabla_latex(df, columnas, encabezado, titulo, etiqueta, formato):
    coma = lambda v, f: (f.format(v)).replace(".", ",") if isinstance(v, (float, np.floating)) else str(v)
    lineas = [r"\begin{table}[htbp]", r"\centering", r"\small", r"\caption{" + titulo + "}",
              r"\label{" + etiqueta + "}", r"\resizebox{\textwidth}{!}{%",
              r"\begin{tabular}{l" + "c" * (len(columnas) - 1) + "}", r"\hline", encabezado, r"\hline"]
    for _, r in df.iterrows():
        lineas.append(" & ".join(coma(r[c], formato.get(c, "{}")) for c in columnas) + r" \\")
    lineas += [r"\hline", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lineas)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx", nargs="?", default="Compilacion_temperatura_compost.xlsx")
    ap.add_argument("--salida", default="salida_etapa3")
    args = ap.parse_args()
    out = Path(args.salida)
    (out / "figuras").mkdir(parents=True, exist_ok=True)

    series, _, _ = cargar_conjunto(args.xlsx)
    etiquetadas = [s for s in series if s.etiquetada]
    print(f"Series: {len(series)} | etiquetadas: {len(etiquetadas)} "
          f"({sum(int(np.sum(s.fase_ref != None)) for s in etiquetadas)} días)")  # noqa: E711

    # 1. LOO sobre las series etiquetadas
    pred = evaluar_loo(etiquetadas)
    nombres = list(tecnicas(None).keys())
    loo = resumen_loo(etiquetadas, pred, nombres).round(3)
    loo.to_csv(out / "evaluacion_loo.csv", index=False)
    print("\n1) Series etiquetadas (dejando una fuera):\n", loo.to_string(index=False))

    # 3. sensibilidad: número fijo de tramos y umbral de la derivada
    from compostaje.identificadores import n_cambios_por_derivada
    sens = []
    for k in N_TRAMOS:
        r = evaluar_loo_una(etiquetadas, lambda m, k=k: IdentificadorHibrido(m, RADIO, k))
        sens.append(dict(criterio="fijo", parametro=k, tramos_medios=k, exactitud=r.exactitud,
                         f1_macro=r.f1_macro, error_medio_dias=r.error_medio_dias))
    for u in UMBRALES_DERIVADA:
        r = evaluar_loo_una(etiquetadas, lambda m, u=u: IdentificadorHibrido(
            m, RADIO, "derivada", umbral_derivada=u))
        tramos = np.mean([n_cambios_por_derivada(s.temperaturas, u) + 1 for s in series])
        sens.append(dict(criterio="derivada", parametro=u, tramos_medios=tramos, exactitud=r.exactitud,
                         f1_macro=r.f1_macro, error_medio_dias=r.error_medio_dias))
    sens = pd.DataFrame(sens).round(3)
    sens.to_csv(out / "sensibilidad_tramos.csv", index=False)
    print("\n3) Híbrido según el número de tramos:\n", sens.to_string(index=False))
    fig_sensibilidad(sens, out / "figuras" / "sensibilidad_tramos.png")

    # 2. todas las series: modelo final con todas las etiquetadas;
    #    las etiquetadas usan su predicción LOO para no evaluarse con sus propios datos.
    modelo = ModeloFases(VENTANA).entrenar(etiquetadas)
    resultados = {n: {} for n in nombres}
    tiempos = {n: [] for n in nombres}
    for s in series:
        for n, tec in tecnicas(modelo).items():
            t0 = time.perf_counter()
            r = tec.identificar(s)
            tiempos[n].append(time.perf_counter() - t0)
            resultados[n][s.nombre] = pred.get((n, s.nombre), r)
    coh = coherencia(resultados, series).round(1)
    coh["seg_por_serie"] = [np.mean(tiempos[n]) for n in coh.tecnica]
    coh["seg_max"] = [np.max(tiempos[n]) for n in coh.tecnica]
    coh = coh.round(3)
    coh.to_csv(out / "coherencia_y_tiempos.csv", index=False)
    print("\n2) Coherencia en todas las series y tiempo por serie:\n", coh.to_string(index=False))

    filas = []
    for s in series:
        for n in nombres:
            filas.append(pd.DataFrame(dict(serie_id=s.nombre, tecnica=n, dia=s.tiempos.astype(int),
                                           temp_suav_c=s.temperaturas.round(2),
                                           fase_ref=s.fase_ref if s.fase_ref is not None else None,
                                           fase=resultados[n][s.nombre].fase_por_dia)))
    pd.concat(filas).to_csv(out / "fases_por_dia.csv", index=False)

    # Figuras
    fig_antes_despues(series, resultados, ["PILA_8", "PILA_11", "PILA_23", "PILA_16", "PILA_15"],
                      out / "figuras" / "antes_despues_campo.png", [nombres[0], nombres[1], FINAL])
    fig_antes_despues(etiquetadas, resultados, [s.nombre for s in etiquetadas],
                      out / "figuras" / "antes_despues_etiquetadas.png", [nombres[0], nombres[1], FINAL])
    fig_todas(series, resultados, FINAL, out / "figuras" / "hibrido_todas.png")

    # Tablas LaTeX
    fmt2 = {c: "{:.2f}" for c in loo.columns}
    fmt2.update({c: "{:.1f}" for c in loo.columns if c.startswith("error")})
    (out / "tabla_loo.tex").write_text(tabla_latex(
        loo, ["tecnica", "exactitud", "f1_macro", "f1_mesófila", "f1_termófila", "f1_enfriamiento",
              "f1_maduración", "error_medio_dias"],
        r"Técnica & Exactitud & F1 macro & Mes. & Term. & Enfr. & Mad. & Error (días) \\",
        "Comparación antes y después de la mejora sobre las series etiquetadas "
        "(validación dejando una serie fuera).", "tab:antes_despues_loo", fmt2), encoding="utf-8")
    fmt1 = {"pct_coherentes": "{:.0f}", "seg_por_serie": "{:.2f}"}
    (out / "tabla_coherencia.tex").write_text(tabla_latex(
        coh, ["tecnica", "coherentes", "pct_coherentes", "inician_mesofila", "llegan_maduracion",
              "transiciones_invalidas", "termofila_bajo_40", "seg_por_serie"],
        r"Técnica & Coherentes & \% & Inician mesófila & Llegan a maduración & Trans. inválidas & Termófila $<$40\,°C & s/serie \\",
        f"Coherencia de la secuencia de fases en las {len(series)} series del conjunto.",
        "tab:coherencia", fmt1), encoding="utf-8")
    print(f"\nResultados en '{out}/'.")


if __name__ == "__main__":
    main()

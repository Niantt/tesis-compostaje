"""
bootstrap_oob.py — Tesis INFO-1197, Etapa 3
Validación bootstrap con curvas fuera de la bolsa (OOB), según el protocolo del
profesor guía (retroalimentación de la Etapa 2):

  1. De las N curvas etiquetadas se sortean N con reemplazo -> entrenamiento.
  2. Las curvas no sorteadas (OOB) se reservan para evaluar. El porcentaje de
     prueba varía entre repeticiones.
  3. Las curvas sorteadas (distintas) se dividen en dos grupos; todas las copias
     de una curva quedan en el mismo grupo:
       grupo A -> construye la plantilla modelo de cada fase (DBA, largo típico de la fase)
       grupo B -> genera las características DTW y entrena KNN
  4. Los fragmentos se generan después de la separación. Cada fragmento termina
     en el día que clasifica y usa solo temperaturas hasta ese día (filtro causal).
  5. Se evalúa en las curvas OOB.
  6. Se repite R veces. Las repeticiones sin curvas suficientes (sin OOB, o con
     menos de 2 curvas distintas para formar los grupos) se registran como no
     evaluables.

Variantes de KNN evaluadas en las mismas particiones:
  largos de fragmento = (3,), (7,), (3, 7, 14), (3, 7, 14, 21)  ·  k = 1, 3, 5
  y la última con el día transcurrido desde el inicio como característica extra
  (es un dato causal: se conoce en cada día).
Referencia en las mismas particiones: DTW de la Etapa 2 (ventana centrada de 21
días, 1-NN), entrenado con todas las curvas sorteadas. No es causal.

Uso: python experimentos/bootstrap_oob.py [ruta.xlsx] [--repeticiones 1000] [--filtro causal]
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compostaje import FASES, IdentificadorDTW, ModeloFases, Preprocesador  # noqa: E402
from compostaje.compilacion import cargar_conjunto  # noqa: E402
from compostaje.knn_dtw import PlantillasFase  # noqa: E402

warnings.filterwarnings("ignore")
LARGOS = (3, 7, 14, 21)
VARIANTES_LARGO = {"L=3": (3,), "L=7": (7,), "L=3,7,14": (3, 7, 14),
                   "L=3,7,14,21": (3, 7, 14, 21)}
KS = (1, 3, 5)
FILTROS = {"causal": dict(filtro="mediana_media", ventana_mediana=5, ventana_media=3, causal=True),
           "savgol": dict(filtro="savgol")}


def columnas(variante):
    """Índices de columnas de la matriz de características para una variante de largos."""
    return [LARGOS.index(L) * len(FASES) + j for L in VARIANTES_LARGO[variante] for j in range(len(FASES))]


def metricas(y, p):
    f1s = f1_score(y, p, labels=FASES, average=None, zero_division=0)
    return dict(f1_macro=f1s.mean(), exactitud=float(np.mean(y == p)),
                **{f"f1_{f.lower()}": v for f, v in zip(FASES, f1s)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx", nargs="?", default="Compilacion_temperatura_compost.xlsx")
    ap.add_argument("--repeticiones", type=int, default=1000)
    ap.add_argument("--filtro", choices=list(FILTROS), default="causal")
    ap.add_argument("--semilla", type=int, default=2026)
    ap.add_argument("--salida", default="salida_etapa3")
    args = ap.parse_args()
    out = Path(args.salida)
    (out / "figuras").mkdir(parents=True, exist_ok=True)

    series, _, _ = cargar_conjunto(args.xlsx, Preprocesador(**FILTROS[args.filtro]))
    curvas = [s for s in series if s.etiquetada]
    N = len(curvas)
    rng = np.random.default_rng(args.semilla)
    print(f"Curvas etiquetadas: {N} | repeticiones: {args.repeticiones} | filtro: {args.filtro}")

    registro, filas = [], []
    t0 = time.perf_counter()
    for rep in range(args.repeticiones):
        sorteo = rng.integers(0, N, N)                              # 1. con reemplazo
        copias = np.bincount(sorteo, minlength=N)
        distintas = np.flatnonzero(copias)
        oob = np.flatnonzero(copias == 0)                           # 2. fuera de la bolsa
        info = dict(rep=rep, distintas=len(distintas), oob=len(oob),
                    pct_prueba=100 * len(oob) / N, evaluable=True, motivo="")
        if len(oob) == 0 or len(distintas) < 2:
            info.update(evaluable=False,
                        motivo="sin curvas OOB" if len(oob) == 0 else "menos de 2 curvas distintas")
            registro.append(info)
            continue
        orden = rng.permutation(distintas)                          # 3. grupos A y B
        n_a = len(orden) // 2
        grupo_a, grupo_b = orden[:n_a], orden[n_a:]
        info.update(grupo_a=",".join(curvas[i].nombre for i in grupo_a),
                    grupo_b=",".join(curvas[i].nombre for i in grupo_b),
                    prueba=",".join(curvas[i].nombre for i in oob))
        registro.append(info)

        # plantillas con el grupo A (cada copia cuenta)
        plant = PlantillasFase().entrenar([curvas[i] for i in grupo_a for _ in range(copias[i])])
        # 4. características DTW (fragmentos causales) del grupo B y de las OOB
        feats = {i: plant.caracteristicas(curvas[i].temperaturas, LARGOS)
                 for i in np.concatenate([grupo_b, oob])}
        y_oob = np.concatenate([curvas[i].fase_ref for i in oob]).astype(str)

        for var, con_dia in [(v, False) for v in VARIANTES_LARGO] + [("L=3,7,14,21", True)]:
            cols = columnas(var)
            f = (lambda i: np.column_stack([feats[i][:, cols], curvas[i].tiempos])) if con_dia \
                else (lambda i: feats[i][:, cols])
            Xb = np.vstack([f(i) for i in grupo_b for _ in range(copias[i])])
            yb = np.concatenate([curvas[i].fase_ref for i in grupo_b for _ in range(copias[i])]).astype(str)
            esc = StandardScaler().fit(Xb)
            Xo = esc.transform(np.vstack([f(i) for i in oob]))
            for k in KS:
                knn = KNeighborsClassifier(n_neighbors=min(k, len(yb))).fit(esc.transform(Xb), yb)
                filas.append(dict(rep=rep, metodo=f"KNN-DTW {var}{' + día' if con_dia else ''}, k={k}",
                                  **metricas(y_oob, knn.predict(Xo))))

        # referencia: DTW de la Etapa 2 con todas las curvas sorteadas
        base = IdentificadorDTW(ModeloFases(21).entrenar([curvas[i] for i in distintas]))
        p = np.concatenate([base.identificar(curvas[i]).fase_por_dia for i in oob]).astype(str)
        filas.append(dict(rep=rep, metodo="DTW Etapa 2 (21 d, centrada)", **metricas(y_oob, p)))

        if (rep + 1) % 50 == 0:
            print(f"  {rep + 1} repeticiones · {time.perf_counter() - t0:.0f} s")

    reg = pd.DataFrame(registro)
    res = pd.DataFrame(filas)
    sufijo = f"_{args.filtro}"
    reg.to_csv(out / f"bootstrap_registro{sufijo}.csv", index=False)
    res.to_csv(out / f"bootstrap_resultados{sufijo}.csv", index=False)

    ev = int(reg.evaluable.sum())
    print(f"\nEvaluables: {ev} de {len(reg)} · no evaluables: "
          f"{reg.loc[~reg.evaluable, 'motivo'].value_counts().to_dict()}")
    print(f"Curvas OOB por repetición evaluable: {reg.loc[reg.evaluable, 'oob'].value_counts().sort_index().to_dict()}"
          f" · % de prueba medio: {reg.loc[reg.evaluable, 'pct_prueba'].mean():.0f} %")

    cols = ["f1_macro"] + [f"f1_{f.lower()}" for f in FASES] + ["exactitud"]
    resumen = []
    for m, g in res.groupby("metodo", sort=False):
        fila = dict(metodo=m, repeticiones=len(g))
        for c in cols:
            fila[f"{c}_media"] = g[c].mean()
            fila[f"{c}_p2.5"] = g[c].quantile(0.025)
            fila[f"{c}_p97.5"] = g[c].quantile(0.975)
        resumen.append(fila)
    resumen = pd.DataFrame(resumen).sort_values("f1_macro_media", ascending=False).round(3)
    resumen.to_csv(out / f"bootstrap_resumen{sufijo}.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n", resumen[["metodo", "f1_macro_media", "f1_macro_p2.5", "f1_macro_p97.5",
                         "f1_mesófila_media", "f1_termófila_media", "f1_enfriamiento_media",
                         "f1_maduración_media"]].to_string(index=False))

    # figura: distribución del F1 macro y por fase (mejor KNN frente a la referencia)
    mejor = resumen[resumen.metodo.str.startswith("KNN")].iloc[0].metodo
    ref = "DTW Etapa 2 (21 d, centrada)"
    fig, axs = plt.subplots(1, 5, figsize=(20, 3.6), sharey=True)
    for ax, c in zip(axs, cols[:5]):
        for m, color in [(ref, "#888888"), (mejor, "#de2d26")]:
            ax.hist(res.loc[res.metodo == m, c], bins=np.linspace(0, 1, 26), alpha=0.6, color=color, label=m)
        ax.set_title(c.replace("f1_", "F1 ").replace("macro", "macro"))
        ax.set_xlabel("F1")
    axs[0].set_ylabel("Repeticiones")
    axs[0].legend(fontsize=7)
    fig.suptitle(f"Bootstrap OOB: {ev} repeticiones evaluables de {len(reg)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "figuras" / f"bootstrap_oob{sufijo}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

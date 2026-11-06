"""
plantillas_adaptativas.py — Tesis INFO-1197, Etapa 3
Modelo por fase con subcurvas de largo adaptado a la duración de cada fase
(retroalimentación de la Etapa 2) y comparación con la ventana fija de 21 días.

  1. Evaluación dejando una serie fuera (LOO) de las configuraciones:
       delta (subcurvas de mesófila y termófila) = 2 o 3 días
       ventana larga (enfriamiento y maduración) = 7, 11, 15 o 21 días
       modelo = 1-NN sobre subcurvas o plantilla modelo DBA
     usadas por DTW solo y por el híbrido.
La plantilla modelo con el largo real de cada fase y su comparación con las 5
series etiquetadas están en plantillas_modelo.py.

Uso: python experimentos/plantillas_adaptativas.py [ruta.xlsx] [--salida salida_etapa3]
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tslearn.metrics import cdist_dtw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compostaje import (FASES, IdentificadorDTW, IdentificadorHibrido,  # noqa: E402
                        ModeloFases, ModeloFasesAdaptativo)
from compostaje.compilacion import cargar_conjunto  # noqa: E402

warnings.filterwarnings("ignore")
RADIO = 2
DELTAS = [2, 3]
VENTANAS_LARGAS = [7, 11, 15, 21]


class ModeloPlantillaDBA:
    """Envuelve un ModeloFasesAdaptativo: la referencia de cada fase es su plantilla DBA."""

    def __init__(self, base: ModeloFasesAdaptativo):
        self.base, self.T = base, base.plantillas()
        self.ventana = base.ventana

    def distancias(self, serie, radio):
        return np.column_stack([
            cdist_dtw(self.base.ventanas_consulta(serie, f), self.T[f][None],
                      global_constraint="sakoe_chiba", sakoe_chiba_radius=radio)[:, 0]
            / np.sqrt(self.base.largo[f]) for f in FASES])


def metricas(y_real, y_pred):
    f1s = f1_score(y_real, y_pred, labels=FASES, average=None, zero_division=0)
    return dict(exactitud=accuracy_score(y_real, y_pred), f1_macro=f1s.mean(),
                **{f"f1_{f.lower()}": v for f, v in zip(FASES, f1s)})


def evaluar(etiquetadas, fabrica, tecnica):
    """LOO: fabrica(series de entrenamiento) -> modelo; tecnica(modelo) -> identificador."""
    y_real, y_pred, por_serie = [], [], {}
    for s in etiquetadas:
        modelo = fabrica([o for o in etiquetadas if o is not s])
        r = tecnica(modelo).identificar(s)
        y_real += list(s.fase_ref)
        y_pred += list(r.fase_por_dia)
        por_serie[s.nombre] = r
    return metricas(y_real, y_pred), por_serie


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx", nargs="?", default="Compilacion_temperatura_compost.xlsx")
    ap.add_argument("--salida", default="salida_etapa3")
    args = ap.parse_args()
    out = Path(args.salida)
    (out / "figuras").mkdir(parents=True, exist_ok=True)

    series, _, _ = cargar_conjunto(args.xlsx)
    etiquetadas = [s for s in series if s.etiquetada]

    # 1. comparación de configuraciones (LOO)
    filas, por_serie = [], {}
    configs = [("Ventana fija 21 d (Etapa 2)", None, 21, "1-NN",
                lambda tr: ModeloFases(21).entrenar(tr))]
    for d in DELTAS:
        for w in VENTANAS_LARGAS:
            configs.append((f"Adaptativa δ={d}, W={w}", d, w, "1-NN",
                            lambda tr, d=d, w=w: ModeloFasesAdaptativo(d, w).entrenar(tr)))
            configs.append((f"Adaptativa δ={d}, W={w}", d, w, "Plantilla DBA",
                            lambda tr, d=d, w=w: ModeloPlantillaDBA(ModeloFasesAdaptativo(d, w).entrenar(tr))))
    for nombre, d, w, modelo, fabrica in configs:
        for tec_nombre, tec in [("DTW", lambda m: IdentificadorDTW(m, RADIO)),
                                ("Híbrido", lambda m: IdentificadorHibrido(m, RADIO))]:
            m, p = evaluar(etiquetadas, fabrica, tec)
            filas.append(dict(configuracion=nombre, delta=d, ventana_larga=w, modelo=modelo,
                              tecnica=tec_nombre, **m))
            por_serie[(nombre, modelo, tec_nombre)] = p
            print(f"  {nombre:<28} {modelo:<13} {tec_nombre:<8} "
                  f"exactitud={m['exactitud']:.3f}  F1 macro={m['f1_macro']:.3f}")
    ev = pd.DataFrame(filas).round(3)
    ev.to_csv(out / "eval_plantillas_adaptativas.csv", index=False)

    # mejor configuración adaptativa con DTW solo (criterio de la Etapa 2: F1 macro)
    adapt = ev[(ev.delta.notna()) & (ev.tecnica == "DTW")].sort_values("f1_macro", ascending=False)
    mejor = adapt.iloc[0]
    d, w = int(mejor.delta), int(mejor.ventana_larga)
    print(f"\nMejor adaptativa (DTW): δ={d}, W={w}, {mejor.modelo} -> F1 {mejor.f1_macro:.3f}")
    base = ev[ev.delta.isna()]
    print("\nVentana fija de 21 días:\n", base[["tecnica", "exactitud", "f1_macro"]].to_string(index=False))

    # resultados por serie: fija vs mejor adaptativa
    filas = []
    for etiqueta, clave in [("Fija 21 d", ("Ventana fija 21 d (Etapa 2)", "1-NN", "DTW")),
                            (f"Adaptativa δ={d}, W={w}", (mejor.configuracion, mejor.modelo, "DTW"))]:
        for s in etiquetadas:
            r = por_serie[clave][s.nombre]
            filas.append(dict(modelo=etiqueta, serie_id=s.nombre, **metricas(s.fase_ref, r.fase_por_dia)))
    ps = pd.DataFrame(filas).round(3)
    ps.to_csv(out / "eval_por_serie_fija_vs_adaptativa.csv", index=False)
    print("\nPor serie:\n", ps[["modelo", "serie_id", "exactitud", "f1_macro"]].to_string(index=False))


if __name__ == "__main__":
    main()

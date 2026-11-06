"""
preprocesamiento.py — Tesis INFO-1197
Carga, inventario y preprocesamiento de las series de temperatura de compostaje
contenidas en "Compilacion_temperatura_compost.xlsx".

Etapas:
  1. Carga: convierte las 5 hojas (formatos distintos) a un formato largo único.
  2. Inventario: una fila por serie (n.º de muestras, duración, intervalos, huecos...).
  3. Preprocesamiento:
       a) remuestreo a periodo fijo de 1 día (series horarias -> promedio diario)
       b) interpolación PCHIP sobre la grilla diaria (no genera sobreoscilaciones)
       c) suavizado Savitzky-Golay (polinomios locales) para eliminar ruido
       d) normalización min-max (solo para DTW; change-point usa °C reales)
  4. Figuras de apoyo para el informe.

Uso:
  python preprocesamiento.py ruta/al/archivo.xlsx  [--salida carpeta]
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

# ---------------------------------------------------------------------------
# Parámetros del preprocesamiento (documentar en el informe)
# ---------------------------------------------------------------------------
PERIODO_DIAS = 1          # periodo de remuestreo
SG_VENTANA = 7            # ventana Savitzky-Golay en días (impar)
SG_VENTANA_CORTA = 3      # ventana para series < 30 días (evita aplanar el pico termófilo)
SG_ORDEN = 2              # orden del polinomio local
HUECO_LARGO_DIAS = 7      # huecos mayores se marcan como "interpolación poco confiable"
GRADO_POLI_GLOBAL = 6     # solo para la figura comparativa (polinomio global vs S-G)

# Criterios de inclusión (sección "Selección del conjunto de datos" del informe)
T_INICIAL_MAX = 40        # C1: el registro debe partir en fase mesófila (< 40 °C)
DURACION_MIN = 40         # C2: debe cubrir el proceso completo (no solo la termófila)
T_MAX_MIN = 40            # C3: el proceso debe alcanzar la fase termófila (>= 40 °C)
HUECO_MAX = 7             # C4: sin huecos de registro mayores a 7 días

MESES = {"ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6, "jul": 7,
         "ago": 8, "sep": 9, "oct": 10, "nov": 11, "dic": 12}


# ---------------------------------------------------------------------------
# 1. CARGA
# ---------------------------------------------------------------------------
def _parsear_fecha(valor, fecha_previa):
    """Acepta datetime o textos tipo '15 de May' (toma el año de la fila anterior)."""
    if hasattr(valor, "year"):
        return pd.Timestamp(valor)
    if isinstance(valor, str):
        m = re.match(r"\s*(\d{1,2})\s*de\s*([a-zA-Záéíóú]+)", valor)
        if m and fecha_previa is not None:
            mes = MESES.get(m.group(2).lower()[:3])
            if mes:
                return pd.Timestamp(fecha_previa.year, mes, int(m.group(1)))
    return None


def _a_numero(serie):
    """Convierte a float aceptando coma decimal y marcadores como '--'."""
    return pd.to_numeric(serie.astype(str).str.replace(",", ".").str.strip(),
                         errors="coerce")


def cargar_pilas_campo(hojas):
    """Hojas PARTE 1-3: composteras Pillanlelbun, registro manual cada 1-6 días.
    Debajo de algunas pilas (PARTE 2) hay tablas de diagnóstico: se extraen aparte."""
    filas, notas = [], []
    for nombre_hoja, campana in [("PARTE 1", 2026), ("PARTE 2", 2025), ("PARTE 3", 2024)]:
        h = hojas[nombre_hoja]
        for c in range(1, h.shape[1] - 1, 2):
            pila = h.iat[2, c]
            if not (isinstance(pila, str) and pila.upper().startswith("PILA")):
                continue
            sid = pila.strip().upper().replace(" ", "_")
            fecha_prev, r = None, 4
            # bloque de datos: hasta la primera fila sin fecha
            while r < h.shape[0] and pd.notna(h.iat[r, c]):
                f = _parsear_fecha(h.iat[r, c], fecha_prev)
                t = pd.to_numeric(h.iat[r, c + 1], errors="coerce")
                if f is not None and pd.notna(t):
                    filas.append(dict(serie_id=sid, fuente="Pillanlelbun", campana=campana,
                                      tipo_registro="manual puntual", fecha=f, temp_c=float(t)))
                    fecha_prev = f
                r += 1
            # tabla de diagnóstico (si existe) más abajo en la misma columna
            for rr in range(r, h.shape[0]):
                if str(h.iat[rr, c]).strip().lower() == "fecha":
                    for k in range(rr + 1, h.shape[0]):
                        if pd.isna(h.iat[k, c]) and pd.isna(h.iat[k, c + 1]):
                            break
                        notas.append(dict(serie_id=sid, fecha_o_rango=str(h.iat[k, c]).strip(),
                                          diagnostico=str(h.iat[k, c + 1]).strip()))
                    break
    return filas, notas


def _etiquetas_desde_columna(col_etiqueta, col_dia, fila_ini, fila_fin):
    """Lee una columna con nombres de fase en la fila donde la fase comienza."""
    etiquetas = []
    for r in range(fila_ini, fila_fin):
        v = col_etiqueta.iat[r]
        if isinstance(v, str) and v.strip():
            fase = re.sub(r"^M\d\s*", "", v.strip()).capitalize()
            etiquetas.append((fase, float(col_dia.iat[r])))
    return etiquetas


def cargar_tesis_y_proyectos(hojas):
    filas, etiquetas = [], []

    # --- PARTE 4: tesis doctoral (3 pilas, 120 días, promedio diario, con fases) ---
    h = hojas["PARTE 4"]
    fases = _etiquetas_desde_columna(h[1], h[2], 3, h.shape[0])
    for c in (3, 4, 5):
        sid = "DOCTORAL_" + str(h.iat[2, c]).strip().upper().replace(" ", "_")
        for r in range(3, h.shape[0]):
            d, t = pd.to_numeric(h.iat[r, 2], errors="coerce"), pd.to_numeric(h.iat[r, c], errors="coerce")
            if pd.notna(d) and pd.notna(t):
                filas.append(dict(serie_id=sid, fuente="Tesis doctoral", tipo_registro="promedio diario",
                                  dia=float(d), temp_c=float(t)))
        etiquetas += [dict(serie_id=sid, fase=f, dia_inicio=d) for f, d in fases]

    # --- PARTE 4: proyecto VIU (solo fase termófila, 11 días) ---
    for r in range(3, h.shape[0]):
        d, t = pd.to_numeric(h.iat[r, 8], errors="coerce"), pd.to_numeric(h.iat[r, 9], errors="coerce")
        if pd.notna(d) and pd.notna(t):
            filas.append(dict(serie_id="VIU", fuente="Proyecto VIU", tipo_registro="promedio diario",
                              dia=float(d), temp_c=float(t)))

    # --- PARTE 5: Quintrecura (registro horario de sensor, 3 series) ---
    h = hojas["PARTE 5"]
    crudo = {}
    for c, nombre in [(0, "T0"), (2, "T1"), (4, "T2")]:
        df = pd.DataFrame({"fecha": pd.to_datetime(h.iloc[6:, c], errors="coerce"),
                           "temp_c": _a_numero(h.iloc[6:, c + 1])}).dropna()
        # descarta marcas de tiempo aisladas fuera de orden (error del datalogger)
        prev, sig = df["fecha"].shift(1), df["fecha"].shift(-1)
        crudo[nombre] = df[(prev.isna() | (df["fecha"] > prev)) & (sig.isna() | (df["fecha"] < sig))]
    # Problema detectado: desde 2025-02-27 18:09 la columna T1 es una copia exacta de T2
    # (misma marca de tiempo y valor). Se recorta esa cola copiada de T1.
    t1 = crudo["T1"].join(crudo["T2"], rsuffix="_t2", how="left")
    igual = (t1["fecha"] == t1["fecha_t2"]) & (t1["temp_c"] == t1["temp_c_t2"])
    cola = igual[::-1].cumprod()[::-1].astype(bool)          # True solo en la cola final idéntica
    if cola.sum() > 24:
        crudo["T1"] = crudo["T1"][~cola]
    for nombre, df in crudo.items():
        for _, row in df.iterrows():
            filas.append(dict(serie_id=f"QUINTRECURA_{nombre}", fuente="Tesis Quintrecura",
                              tipo_registro="horario (sensor)", fecha=row.fecha, temp_c=row.temp_c))

    # --- PARTE 5: Peñailillo (3 tratamientos, 90 días, con fases) ---
    fases = _etiquetas_desde_columna(h[7], h[8], 6, h.shape[0])
    for c in (9, 10, 11):
        sid = "PENAILILLO_" + str(h.iat[5, c]).strip().upper().replace("-", "_")
        for r in range(6, h.shape[0]):
            d = pd.to_numeric(h.iat[r, 8], errors="coerce")
            t = _a_numero(pd.Series([h.iat[r, c]])).iat[0]
            if pd.notna(d) and pd.notna(t):
                filas.append(dict(serie_id=sid, fuente="Tesis Peñailillo", tipo_registro="diario",
                                  dia=float(d), temp_c=float(t)))
        etiquetas += [dict(serie_id=sid, fase=f, dia_inicio=d) for f, d in fases]

    # --- PARTE 5: Opazo (6 series, solo etapa termófila, 20 días) ---
    for c in range(14, 20):
        nombre = re.sub(r"[^A-Z0-9]+", "_", str(h.iat[5, c]).upper()).strip("_")
        for r in range(6, h.shape[0]):
            d, t = pd.to_numeric(h.iat[r, 13], errors="coerce"), pd.to_numeric(h.iat[r, c], errors="coerce")
            if pd.notna(d) and pd.notna(t):
                filas.append(dict(serie_id=f"OPAZO_{nombre}", fuente="Tesis Opazo",
                                  tipo_registro="promedio diario", dia=float(d), temp_c=float(t)))
    return filas, etiquetas


def cargar_todo(ruta_xlsx):
    hojas = pd.read_excel(ruta_xlsx, sheet_name=None, header=None)
    f1, notas = cargar_pilas_campo(hojas)
    f2, etiquetas = cargar_tesis_y_proyectos(hojas)
    df = pd.DataFrame(f1 + f2)
    # tiempo relativo en días desde la primera muestra de cada serie
    con_fecha = df["fecha"].notna()
    t0 = df[con_fecha].groupby("serie_id")["fecha"].transform("min")
    df.loc[con_fecha, "dia"] = (df.loc[con_fecha, "fecha"] - t0).dt.total_seconds() / 86400 + 1
    df["t_dias"] = df.groupby("serie_id")["dia"].transform(lambda s: s - s.min())
    # orden natural (PILA_2 antes que PILA_10)
    df["_orden"] = df["serie_id"].map(lambda s: [int(t) if t.isdigit() else t
                                                 for t in re.split(r"(\d+)", s)])
    df = (df.sort_values(["fuente", "t_dias"]).sort_values("_orden", kind="stable")
            .drop(columns="_orden").reset_index(drop=True))
    return df, pd.DataFrame(notas), pd.DataFrame(etiquetas)


# ---------------------------------------------------------------------------
# 2. INVENTARIO
# ---------------------------------------------------------------------------
SOLO_TERMOFILA = {"Proyecto VIU", "Tesis Opazo"}


def inventario(df, etiquetas):
    filas = []
    for sid, g in df.groupby("serie_id", sort=False):
        dt = g["t_dias"].diff().dropna()
        fuente = g["fuente"].iat[0]
        filas.append(dict(
            serie_id=sid, fuente=fuente, tipo_registro=g["tipo_registro"].iat[0],
            n_muestras=len(g),
            duracion_dias=round(g["t_dias"].max(), 1),
            intervalo_mediano_dias=round(dt.median(), 3) if len(dt) else np.nan,
            hueco_max_dias=round(dt.max(), 1) if len(dt) else np.nan,
            t_inicial_c=round(g["temp_c"].iat[0], 1),
            t_max_c=round(g["temp_c"].max(), 1),
            dia_t_max=round(g.loc[g["temp_c"].idxmax(), "t_dias"], 1),
            alcanza_40c=bool((g["temp_c"] >= 40).any()),
            alcanza_55c=bool((g["temp_c"] >= 55).any()),
            cobertura=("solo termófila" if fuente in SOLO_TERMOFILA else
                       "inicia en termófila" if g["temp_c"].iat[0] >= T_INICIAL_MAX else "proceso completo"),
            etiquetas_fase=sid in set(etiquetas["serie_id"]),
        ))
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# 3. PREPROCESAMIENTO
# ---------------------------------------------------------------------------
def preprocesar_serie(g):
    """Devuelve la serie en grilla diaria: interpolada (PCHIP), suavizada (S-G) y normalizada."""
    # a) series horarias -> promedio diario
    if g["tipo_registro"].iat[0].startswith("horario"):
        g = (g.assign(dia_ent=np.floor(g["t_dias"]))
               .groupby("dia_ent", as_index=False)["temp_c"].mean()
               .rename(columns={"dia_ent": "t_dias"}))
    else:
        g = g.groupby("t_dias", as_index=False)["temp_c"].mean()  # fusiona duplicados

    x, y = g["t_dias"].to_numpy(), g["temp_c"].to_numpy()
    grilla = np.arange(0, np.floor(x.max()) + 1, PERIODO_DIAS)

    # b) interpolación PCHIP
    y_int = PchipInterpolator(x, y)(grilla)

    # longitud del hueco original al que pertenece cada punto de la grilla
    idx = np.clip(np.searchsorted(x, grilla, side="right"), 1, len(x) - 1)
    hueco = np.where(np.isin(grilla, x), 0.0, x[idx] - x[idx - 1])

    # c) suavizado Savitzky-Golay (ventana adaptada si la serie es corta)
    base = SG_VENTANA if len(grilla) >= 30 else SG_VENTANA_CORTA
    ventana = min(base, len(grilla) if len(grilla) % 2 else len(grilla) - 1)
    y_sg = savgol_filter(y_int, ventana, min(SG_ORDEN, ventana - 1)) if ventana >= 3 else y_int

    # d) normalización min-max (para DTW)
    rango = y_sg.max() - y_sg.min()
    y_norm = (y_sg - y_sg.min()) / rango if rango > 0 else np.zeros_like(y_sg)

    return pd.DataFrame(dict(t_dias=grilla, temp_interp_c=y_int.round(3),
                             temp_suav_c=y_sg.round(3), temp_norm=y_norm.round(4),
                             es_medicion=np.isin(grilla, x), hueco_dias=hueco,
                             hueco_largo=hueco > HUECO_LARGO_DIAS))


def asignar_fase(diario, etiquetas):
    """Agrega la fase de referencia (si la serie tiene etiquetas)."""
    diario["fase_ref"] = None
    for sid, e in etiquetas.groupby("serie_id"):
        e = e.sort_values("dia_inicio")
        m = diario["serie_id"] == sid
        dia = diario.loc[m, "t_dias"] + 1  # las etiquetas usan día 1 = inicio
        diario.loc[m, "fase_ref"] = pd.cut(dia, bins=list(e["dia_inicio"]) + [np.inf],
                                           labels=list(e["fase"]), right=False,
                                           ordered=False).astype(object)
    return diario


# ---------------------------------------------------------------------------
# 4. FIGURAS
# ---------------------------------------------------------------------------
def fig_todas(df, diario, ruta):
    ids = list(df["serie_id"].unique())
    ncol = 6
    nfil = int(np.ceil(len(ids) / ncol))
    fig, axs = plt.subplots(nfil, ncol, figsize=(18, 2.3 * nfil), sharey=True)
    for ax, sid in zip(axs.flat, ids):
        crudo, d = df[df.serie_id == sid], diario[diario.serie_id == sid]
        ax.plot(crudo.t_dias, crudo.temp_c, ".", ms=2, color="0.6")
        ax.plot(d.t_dias, d.temp_suav_c, lw=1.2, color="C3")
        ax.axhline(T_MAX_MIN, ls=":", lw=0.8, color="k")
        ax.set_title(sid, fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axs.flat[len(ids):]:
        ax.axis("off")
    fig.supxlabel("Días desde el inicio del registro")
    fig.supylabel("Temperatura (°C)")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def fig_comparacion_suavizado(df, diario, sids, ruta):
    """Polinomio global vs. Savitzky-Golay: justifica la elección del método."""
    fig, axs = plt.subplots(1, len(sids), figsize=(6 * len(sids), 4))
    resumen = []
    for ax, sid in zip(np.atleast_1d(axs), sids):
        crudo, d = df[df.serie_id == sid], diario[diario.serie_id == sid]
        poli = np.polynomial.Polynomial.fit(crudo.t_dias, crudo.temp_c, GRADO_POLI_GLOBAL)
        ax.plot(crudo.t_dias, crudo.temp_c, ".", ms=3, color="0.5", label="medición")
        ax.plot(d.t_dias, d.temp_interp_c, lw=0.8, color="C0", alpha=0.6, label="PCHIP (1 día)")
        ax.plot(d.t_dias, poli(d.t_dias), "--", color="C1", label=f"polinomio global grado {GRADO_POLI_GLOBAL}")
        ax.plot(d.t_dias, d.temp_suav_c, color="C3", lw=1.6, label=f"Savitzky-Golay ({SG_VENTANA} d, orden {SG_ORDEN})")
        ax.set_title(sid)
        ax.set_xlabel("Días")
        ax.set_ylabel("Temperatura (°C)")
        ax.legend(fontsize=8)
        sg_en_med = np.interp(crudo.t_dias, d.t_dias, d.temp_suav_c)
        resumen.append(dict(serie_id=sid,
                            rmse_poli=np.sqrt(np.mean((poli(crudo.t_dias) - crudo.temp_c) ** 2)),
                            rmse_sg=np.sqrt(np.mean((sg_en_med - crudo.temp_c) ** 2))))
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return pd.DataFrame(resumen).round(2)


def fig_etiquetadas(diario, etiquetas, ruta):
    colores = {"Mesófila": "#9ecae1", "Termófila": "#fc9272",
               "Enfriamiento": "#fdd0a2", "Maduración": "#a1d99b"}
    ids = list(etiquetas["serie_id"].unique())
    fig, axs = plt.subplots(2, 3, figsize=(16, 7), sharey=True)
    for ax, sid in zip(axs.flat, ids):
        d = diario[diario.serie_id == sid]
        for fase, col in colores.items():
            dias = d.loc[d.fase_ref == fase, "t_dias"]
            if len(dias):
                ax.axvspan(dias.min(), dias.max() + 1, color=col, alpha=0.6, label=fase)
        ax.plot(d.t_dias, d.temp_suav_c, color="k", lw=1.2)
        ax.set_title(sid)
    for ax in axs.flat[len(ids):]:
        ax.axis("off")
    axs.flat[0].legend(fontsize=8, loc="upper right")
    fig.supxlabel("Días")
    fig.supylabel("Temperatura suavizada (°C)")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# EXPORTACIÓN LaTeX
# ---------------------------------------------------------------------------
def tabla_latex_resumen(inv):
    res = (inv.groupby(["fuente", "tipo_registro", "cobertura"], sort=False)
              .agg(series=("serie_id", "count"), muestras=("n_muestras", "sum"),
                   dur_min=("duracion_dias", "min"), dur_max=("duracion_dias", "max"),
                   intervalo=("intervalo_mediano_dias", "median"),
                   etiquetas=("etiquetas_fase", "all"))
              .reset_index())
    lineas = [r"\begin{table}[htbp]", r"\centering", r"\small",
              r"\caption{Resumen del conjunto de datos de temperatura de compostaje.}",
              r"\label{tab:inventario_datos}",
              r"\resizebox{\textwidth}{!}{%",
              r"\begin{tabular}{llcrrcc}", r"\hline",
              r"Fuente & Tipo de registro & Series & Muestras & Duración (d) & Intervalo (d) & Fases etiq. \\",
              r"\hline"]
    for _, r in res.iterrows():
        dur = f"{r.dur_min:.0f}" if r.dur_min == r.dur_max else f"{r.dur_min:.0f}--{r.dur_max:.0f}"
        intervalo = "1/24" if r.intervalo < 0.5 else f"{r.intervalo:g}"
        nota = {"solo termófila": r"$^{\dagger}$", "inicia en termófila": r"$^{\ddagger}$"}.get(r.cobertura, "")
        lineas.append(f"{r.fuente}{nota} & {r.tipo_registro} & {r.series} & {r.muestras} & "
                      f"{dur} & {intervalo} & {'Sí' if r.etiquetas else 'No'} \\\\")
    lineas += [r"\hline",
               f"Total & & {inv.shape[0]} & {inv.n_muestras.sum()} & & & \\\\",
               r"\hline", r"\end{tabular}}",
               r"\par\smallskip\footnotesize $^{\dagger}$Registran solo la fase termófila. "
               r"$^{\ddagger}$El registro comienza con la pila ya en fase termófila.",
               r"\end{table}"]
    return "\n".join(lineas)


# ---------------------------------------------------------------------------
# SELECCIÓN DEL CONJUNTO DE DATOS
# ---------------------------------------------------------------------------
CRITERIOS = [
    ("C1", f"Inicia en fase mesófila (T inicial < {T_INICIAL_MAX} °C)",
     lambda r: r.t_inicial_c < T_INICIAL_MAX),
    ("C2", f"Registra el proceso completo (duración >= {DURACION_MIN} días)",
     lambda r: r.duracion_dias >= DURACION_MIN),
    ("C3", f"Alcanza la fase termófila (T máx >= {T_MAX_MIN} °C)",
     lambda r: r.t_max_c >= T_MAX_MIN),
    ("C4", f"Sin huecos de registro mayores a {HUECO_MAX} días",
     lambda r: r.hueco_max_dias <= HUECO_MAX),
]


def seleccionar(inv):
    """Aplica los criterios en orden; registra el primero que la serie no cumple."""
    inv = inv.copy()
    motivos = []
    for _, r in inv.iterrows():
        fallo = next((cid for cid, _, regla in CRITERIOS if not regla(r)), "")
        motivos.append(fallo)
    inv["excluida_por"] = motivos
    inv["incluida"] = inv["excluida_por"] == ""
    return inv


def tabla_latex_seleccion(inv):
    """Embudo de selección: cuántas series quedan tras cada criterio."""
    lineas = [r"\begin{table}[htbp]", r"\centering", r"\small",
              r"\caption{Criterios de selección de las series de temperatura.}",
              r"\label{tab:seleccion_datos}",
              r"\resizebox{\textwidth}{!}{%",
              r"\begin{tabular}{clrr}", r"\hline",
              r"Criterio & Descripción & Excluidas & Restantes \\", r"\hline",
              f" & Series disponibles & & {len(inv)} \\\\"]
    restantes = len(inv)
    for cid, desc, _ in CRITERIOS:
        exc = (inv["excluida_por"] == cid).sum()
        restantes -= exc
        lineas.append(f"{cid} & {desc.replace('>=', r'$\geq$')} & {exc} & {restantes} \\\\")
    sel = inv[inv.incluida]
    lineas += [r"\hline",
               r"\multicolumn{4}{l}{Conjunto final: " + f"{len(sel)} series, {sel.n_muestras.sum()} mediciones, "
               + f"{sel.etiquetas_fase.sum()} con fases etiquetadas" + r"} \\",
               r"\hline", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lineas)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx", nargs="?", default="Compilacion_temperatura_compost.xlsx")
    ap.add_argument("--salida", default="salida")
    args = ap.parse_args()
    out = Path(args.salida)
    (out / "figuras").mkdir(parents=True, exist_ok=True)

    df, notas, etiquetas = cargar_todo(args.xlsx)
    inv = seleccionar(inventario(df, etiquetas))
    inv.to_csv(out / "inventario_series.csv", index=False)          # inventario completo (46)
    (out / "tabla_seleccion.tex").write_text(tabla_latex_seleccion(inv).replace("°C", r"\textdegree{}C"),
                                                encoding="utf-8")
    (out / "tabla_inventario.tex").write_text(tabla_latex_resumen(inv), encoding="utf-8")

    # a partir de aquí se trabaja solo con el conjunto seleccionado
    seleccion = inv.loc[inv.incluida, "serie_id"]
    df = df[df.serie_id.isin(seleccion)]
    etiquetas = etiquetas[etiquetas.serie_id.isin(seleccion)]
    notas = notas[notas.serie_id.isin(seleccion)]
    inv_sel = inv[inv.incluida]

    diario = pd.concat([preprocesar_serie(g).assign(serie_id=sid, fuente=g["fuente"].iat[0])
                        for sid, g in df.groupby("serie_id", sort=False)], ignore_index=True)
    diario = asignar_fase(diario, etiquetas)
    diario = diario[["serie_id", "fuente", "t_dias", "temp_interp_c", "temp_suav_c", "temp_norm",
                     "es_medicion", "hueco_dias", "hueco_largo", "fase_ref"]]

    df.to_csv(out / "datos_crudos_largo.csv", index=False)
    diario.to_csv(out / "datos_diarios_preprocesados.csv", index=False)
    etiquetas.to_csv(out / "etiquetas_fase.csv", index=False)
    notas.to_csv(out / "anotaciones_diagnostico.csv", index=False)

    fig_todas(df, diario, out / "figuras" / "series_todas.png")
    rmse = fig_comparacion_suavizado(df, diario, ["PILA_15", "DOCTORAL_PILA_1", "PENAILILLO_C1_SH"],
                                     out / "figuras" / "comparacion_suavizado.png")
    fig_etiquetadas(diario, etiquetas, out / "figuras" / "series_con_fases.png")
    rmse.to_csv(out / "rmse_suavizado.csv", index=False)

    print(f"Disponibles: {len(inv)} series | Seleccionadas: {len(inv_sel)} | "
          f"Mediciones: {len(df)} | Puntos diarios: {len(diario)}")
    print(inv.loc[~inv.incluida, ["serie_id", "excluida_por"]].to_string(index=False))
    print(inv_sel.groupby("fuente", sort=False)["n_muestras"].agg(["count", "sum"]))
    print("\nRMSE respecto a las mediciones:\n", rmse)


if __name__ == "__main__":
    main()

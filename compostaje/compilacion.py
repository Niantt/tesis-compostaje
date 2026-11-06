"""Carga de la compilación de datos del proyecto (Compilacion_temperatura_compost.xlsx).

Convierte las 5 hojas (formatos distintos) a un formato largo único, construye el
inventario de series y aplica los criterios de selección C1-C4. Es el mismo código
de la Etapa 2 (etapa2/preprocesamiento.py), movido al paquete sin cambios de lógica.
"""

import re

import numpy as np
import pandas as pd

# Criterios de inclusión (umbral de 40 °C: Azim et al., 2018; Pérez-San Martín et al., 2025)
T_INICIAL_MAX = 40
DURACION_MIN = 40
T_MAX_MIN = 40
HUECO_MAX = 7

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


def cargar_conjunto(ruta_xlsx, preprocesador=None):
    """Conjunto de trabajo: las series seleccionadas (C1-C4), preprocesadas y con
    sus fases de referencia cuando existen.

    Devuelve (series, inventario, anotaciones_diagnostico).
    """
    from .preprocesador import Preprocesador, asignar_fases_referencia
    from .serie import SerieTemperatura

    pre = preprocesador or Preprocesador()
    df, notas, etiquetas = cargar_todo(ruta_xlsx)
    inv = seleccionar(inventario(df, etiquetas))
    series = []
    for sid in inv.loc[inv.incluida, "serie_id"]:
        g = df[df.serie_id == sid]
        cruda = SerieTemperatura(sid, g["t_dias"].to_numpy(), g["temp_c"].to_numpy(),
                                 metadatos=dict(fuente=g["fuente"].iat[0],
                                                tipo_registro=g["tipo_registro"].iat[0]))
        s = pre.procesar(cruda)
        e = etiquetas[etiquetas.serie_id == sid]
        if len(e):
            asignar_fases_referencia(s, list(zip(e["fase"], e["dia_inicio"])))
        series.append(s)
    return series, inv, notas[notas.serie_id.isin(inv.loc[inv.incluida, "serie_id"])]

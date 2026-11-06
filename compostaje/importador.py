"""Importación de un registro de temperatura desde CSV o XLSX (US-04, RF-01)."""

from pathlib import Path

import warnings

import numpy as np
import pandas as pd

from .serie import SerieTemperatura

_PALABRAS_TIEMPO = ("fecha", "dia", "día", "tiempo", "hora", "time", "date", "day")
_PALABRAS_TEMP = ("temp", "°c", "grados")


class ErrorImportacion(Exception):
    """El archivo no se pudo leer como un registro de temperatura."""


class ImportadorDatos:
    """Lee un archivo con una columna de tiempo y una de temperatura.

    Formato aceptado:
      - Columna de tiempo: fechas (con o sin hora) o días transcurridos.
      - Columna de temperatura en °C. Se acepta coma decimal.
    Si no hay encabezados reconocibles, se usan las dos primeras columnas.
    """

    def leer(self, ruta) -> SerieTemperatura:
        ruta = Path(ruta)
        ext = ruta.suffix.lower()
        if ext == ".csv":
            return self.leer_csv(ruta)
        if ext in (".xlsx", ".xls"):
            return self.leer_xlsx(ruta)
        raise ErrorImportacion(f"Formato no soportado: '{ext}'. Use un archivo CSV o XLSX.")

    def leer_csv(self, ruta) -> SerieTemperatura:
        try:
            df = pd.read_csv(ruta, sep=None, engine="python", encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(ruta, sep=None, engine="python", encoding="latin-1")
        except Exception as e:  # archivo vacío, binario, etc.
            raise ErrorImportacion(f"No se pudo leer el CSV: {e}") from e
        return self._a_serie(df, Path(ruta).stem)

    def leer_xlsx(self, ruta, hoja=0) -> SerieTemperatura:
        try:
            df = pd.read_excel(ruta, sheet_name=hoja)
        except Exception as e:
            raise ErrorImportacion(f"No se pudo leer el XLSX: {e}") from e
        return self._a_serie(df, Path(ruta).stem)

    # ------------------------------------------------------------------
    def _a_serie(self, df: pd.DataFrame, nombre: str) -> SerieTemperatura:
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if df.shape[1] < 2 or len(df) == 0:
            raise ErrorImportacion("El archivo debe tener al menos dos columnas: tiempo y temperatura.")

        cols = [str(c).strip().lower() for c in df.columns]
        c_t = next((i for i, c in enumerate(cols) if any(p in c for p in _PALABRAS_TIEMPO)), 0)
        c_y = next((i for i, c in enumerate(cols)
                    if i != c_t and any(p in c for p in _PALABRAS_TEMP)), 1 if c_t == 0 else 0)

        temp = _a_numero(df.iloc[:, c_y])
        tiempo = df.iloc[:, c_t]
        dias = _a_numero(tiempo)
        if dias.notna().mean() < 0.5:                 # no son números: se interpretan como fechas
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                texto = tiempo.astype(str).str.strip()
                iso = texto.str.match(r"^\d{4}-").mean() > 0.5      # 2025-04-01 (año primero)
                fechas = pd.to_datetime(texto, errors="coerce", format="mixed",
                                        dayfirst=not iso)
            if fechas.notna().mean() < 0.5:
                raise ErrorImportacion("No se reconoce la columna de tiempo (fechas o días).")
            dias = (fechas - fechas.min()).dt.total_seconds() / 86400

        ok = dias.notna() & temp.notna()
        descartadas = int((~ok).sum())
        if ok.sum() < 2:
            raise ErrorImportacion("El archivo no tiene suficientes mediciones numéricas.")
        d = pd.DataFrame({"t": dias[ok].to_numpy(float), "y": temp[ok].to_numpy(float)})
        desordenado = bool((np.diff(d["t"]) < 0).any())
        d = d.sort_values("t", kind="stable")
        return SerieTemperatura(nombre, d["t"] - d["t"].min(), d["y"],
                                metadatos=dict(filas_descartadas=descartadas,
                                               estaba_desordenada=desordenado))


def _a_numero(col: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(col):
        return pd.to_numeric(col, errors="coerce")
    return pd.to_numeric(col.astype(str).str.replace(",", ".").str.strip(), errors="coerce")

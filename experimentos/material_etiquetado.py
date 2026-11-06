"""
material_etiquetado.py — Tesis INFO-1197, Etapa 3
Prepara el material para que el profesor co-guía etiquete las fases de las pilas
de campo (Composteras Pillanlelbun):

  - Etiquetado_pilas.pdf : una página por pila con las mediciones, la curva
                           suavizada y las fechas en el eje horizontal. Sin
                           fases sugeridas, para no influir en su criterio.
  - Planilla_etiquetado.xlsx : una fila por pila para anotar la fecha de inicio de
                           cada fase y observaciones, más una hoja con las
                           mediciones de cada pila.

Uso: python experimentos/material_etiquetado.py [ruta.xlsx] [--salida salida_etapa3/etiquetado]
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compostaje import COLORES  # noqa: E402
from compostaje.compilacion import cargar_conjunto, cargar_todo  # noqa: E402

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx", nargs="?", default="Compilacion_temperatura_compost.xlsx")
    ap.add_argument("--salida", default="salida_etapa3/etiquetado")
    args = ap.parse_args()
    out = Path(args.salida)
    out.mkdir(parents=True, exist_ok=True)

    crudo, notas, _ = cargar_todo(args.xlsx)
    series, _, _ = cargar_conjunto(args.xlsx)
    campo = [s for s in series if s.metadatos.get("fuente") == "Pillanlelbun"]

    filas_resumen, filas_datos = [], []
    with PdfPages(out / "Etiquetado_pilas.pdf") as pdf:
        # portada con instrucciones
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.08, 0.85, "Identificación de fases del compostaje — pilas de campo Pillanlelbun",
                 fontsize=16, weight="bold")
        texto = (
            "Profesor Andrés:\n\n"
            "Este documento muestra la curva de temperatura de cada pila de campo seleccionada para la tesis.\n"
            "Para entrenar y validar el software necesito saber en qué fecha comienza cada fase según su criterio:\n\n"
            "      termófila  ·  enfriamiento  ·  maduración\n\n"
            "Las fechas se anotan en la planilla adjunta (Planilla_etiquetado.xlsx), una fila por pila.\n"
            "Si una pila tiene recalentamientos (por ejemplo, tras un volteo), puede indicarlo en Observaciones\n"
            "con las fechas en que la pila vuelve a fase termófila. Si una fase no se alcanza o no es posible\n"
            "decidirla, basta con dejar la celda vacía o escribir \"no se alcanza\" / \"no es claro\".\n\n"
            "En cada gráfico, los puntos son las mediciones y la línea es la curva suavizada.\n"
            "La línea punteada marca 40 °C. Las curvas no muestran ninguna clasificación del software,\n"
            "para no influir en su criterio.\n\n"
            f"Pilas incluidas: {len(campo)}.\n\nNicolás Escobar Videla")
        fig.text(0.08, 0.12, texto, fontsize=11, va="bottom", linespacing=1.5)
        pdf.savefig(fig)
        plt.close(fig)

        for s in campo:
            c = crudo[crudo.serie_id == s.nombre].sort_values("fecha")
            t0 = c["fecha"].min()
            fechas = t0 + pd.to_timedelta(s.tiempos, unit="D")
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.plot(c["fecha"], c["temp_c"], "o", color="0.35", ms=6, label="Medición", zorder=3)
            ax.plot(fechas, s.temperaturas, color="#de2d26",
                    lw=2, label="Curva suavizada")
            ax.axhline(40, ls=":", color="k", lw=1)
            ax.set_title(f"{s.nombre.replace('_', ' ').title()} — campaña {int(c['campana'].iat[0])}",
                         fontsize=15, weight="bold")
            ax.set_ylabel("Temperatura (°C)", fontsize=12)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
            ax.xaxis.set_minor_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
            plt.setp(ax.get_xticklabels(), rotation=90, fontsize=9)
            ax.grid(which="major", alpha=0.35)
            ax.grid(which="minor", alpha=0.12)
            ax.legend(loc="upper right")
            n = notas[notas.serie_id == s.nombre]
            if len(n):
                txt = "Anotaciones de campo existentes:\n" + "\n".join(
                    f"• {r.fecha_o_rango[:10]}: {r.diagnostico[:110]}" for r in n.itertuples())
                fig.text(0.07, 0.01, txt, fontsize=8, va="bottom")
                fig.subplots_adjust(bottom=0.12 + 0.025 * len(n))
            else:
                fig.subplots_adjust(bottom=0.12)
            pdf.savefig(fig)
            plt.close(fig)
            filas_resumen.append(dict(pila=s.nombre, campana=int(c["campana"].iat[0]),
                                      inicio=t0.date(), fin=c["fecha"].max().date(),
                                      mediciones=len(c), t_max=round(c["temp_c"].max(), 1)))
            for r in c.itertuples():
                filas_datos.append(dict(pila=s.nombre, fecha=r.fecha.date(), temperatura_c=r.temp_c))

    # planilla
    wb = Workbook()
    ws = wb.active
    ws.title = "Etiquetas"
    enc = ["Pila", "Campaña", "Primer registro", "Último registro", "N.º mediciones", "T máx (°C)",
           "Inicio termófila", "Inicio enfriamiento", "Inicio maduración", "Recalentamientos (fechas)",
           "Observaciones"]
    ws.append(enc)
    relleno = {7: COLORES["Termófila"], 8: COLORES["Enfriamiento"], 9: COLORES["Maduración"]}
    for c_idx in range(1, len(enc) + 1):
        celda = ws.cell(1, c_idx)
        celda.font = Font(bold=True)
        celda.alignment = Alignment(wrap_text=True, vertical="center")
        if c_idx in relleno:
            celda.fill = PatternFill("solid", fgColor=relleno[c_idx].lstrip("#"))
    for r in filas_resumen:
        ws.append([r["pila"], r["campana"], r["inicio"], r["fin"], r["mediciones"], r["t_max"],
                   None, None, None, None, None])
    for fila in ws.iter_rows(min_row=2):
        for c_idx in (2, 3):
            fila[c_idx].number_format = "DD-MM-YYYY"
        for c_idx in (6, 7, 8):
            fila[c_idx].number_format = "DD-MM-YYYY"
    anchos = [11, 9, 13, 13, 10, 9, 15, 17, 16, 24, 40]
    for i, w in enumerate(anchos, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.row_dimensions[1].height = 32
    ws.freeze_panes = "B2"

    ws2 = wb.create_sheet("Mediciones")
    ws2.append(["Pila", "Fecha", "Temperatura (°C)"])
    for c_idx in range(1, 4):
        ws2.cell(1, c_idx).font = Font(bold=True)
    for r in filas_datos:
        ws2.append([r["pila"], r["fecha"], r["temperatura_c"]])
    for fila in ws2.iter_rows(min_row=2):
        fila[1].number_format = "DD-MM-YYYY"
    for i, w in enumerate([11, 13, 16], 1):
        ws2.column_dimensions[get_column_letter(i)].width = w
    wb.save(out / "Planilla_etiquetado.xlsx")
    print(f"{len(campo)} pilas -> {out}/Etiquetado_pilas.pdf y Planilla_etiquetado.xlsx")


if __name__ == "__main__":
    main()

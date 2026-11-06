"""Exportación del resultado (RF-07). El reporte PDF completo se termina en la Etapa 4 (US-07)."""

import pandas as pd

from .resultado import ResultadoFases
from .serie import SerieTemperatura


class ExportadorReporte:

    def exportar_csv(self, serie: SerieTemperatura, resultado: ResultadoFases, ruta) -> None:
        pd.DataFrame({"dia": serie.tiempos.astype(int),
                      "temperatura_suavizada_c": serie.temperaturas.round(2),
                      "fase": resultado.fase_por_dia}).to_csv(ruta, index=False, encoding="utf-8-sig")

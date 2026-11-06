"""Validación del formato e integridad de una serie (RF-02)."""

from dataclasses import dataclass

import numpy as np

from .serie import SerieTemperatura

T_MIN_FISICA, T_MAX_FISICA = -10.0, 85.0   # fuera de este rango, error de sensor o digitación
HUECO_MAX_DIAS = 7                          # criterio C4 de la selección de datos
DIAS_MINIMOS = 7                            # bajo esto no se pueden identificar fases


@dataclass
class Aviso:
    nivel: str      # "error" impide continuar; "advertencia" solo informa
    mensaje: str


class ValidadorDatos:

    def validar(self, serie: SerieTemperatura) -> list[Aviso]:
        avisos = []
        t, y = serie.tiempos, serie.temperaturas
        if len(serie) < 2:
            return [Aviso("error", "La serie tiene menos de dos mediciones.")]
        if serie.duracion() < DIAS_MINIMOS:
            avisos.append(Aviso("error", f"La serie dura {serie.duracion():.1f} días; "
                                         f"se necesitan al menos {DIAS_MINIMOS}."))
        fuera = (y < T_MIN_FISICA) | (y > T_MAX_FISICA)
        if fuera.any():
            avisos.append(Aviso("error", f"{int(fuera.sum())} mediciones fuera del rango físico "
                                         f"({T_MIN_FISICA:.0f} a {T_MAX_FISICA:.0f} °C)."))
        rep = int(np.sum(np.diff(np.sort(t)) == 0))
        if rep:
            avisos.append(Aviso("advertencia", f"{rep} mediciones con tiempo repetido; se promediarán."))
        huecos = np.diff(np.sort(np.unique(t)))
        if len(huecos) and huecos.max() > HUECO_MAX_DIAS:
            avisos.append(Aviso("advertencia", f"Hay un hueco de {huecos.max():.0f} días sin registro; "
                                               "la interpolación en ese tramo es poco confiable."))
        if serie.metadatos.get("filas_descartadas"):
            avisos.append(Aviso("advertencia", f"Se descartaron {serie.metadatos['filas_descartadas']} "
                                               "filas sin tiempo o temperatura válidos."))
        if serie.metadatos.get("estaba_desordenada"):
            avisos.append(Aviso("advertencia", "Las mediciones no estaban en orden; se ordenaron por tiempo."))
        if y[0] >= 40:
            avisos.append(Aviso("advertencia", "El registro comienza sobre 40 °C (ya en fase termófila); "
                                               "la fase mesófila no quedará registrada."))
        return avisos

    @staticmethod
    def hay_errores(avisos: list[Aviso]) -> bool:
        return any(a.nivel == "error" for a in avisos)

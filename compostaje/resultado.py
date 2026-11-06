"""Resultado de la identificación de fases."""

from dataclasses import dataclass, field

import numpy as np

from .serie import FASES


@dataclass
class ResultadoFases:
    """Fase asignada a cada día de una serie preprocesada (periodo diario)."""
    tecnica: str
    fase_por_dia: np.ndarray
    tiempos: np.ndarray
    detalles: dict = field(default_factory=dict)

    def __post_init__(self):
        self.fase_por_dia = np.asarray(self.fase_por_dia, dtype=object)
        self.tiempos = np.asarray(self.tiempos, dtype=float)

    def segmentos(self) -> list[tuple[str, float, float]]:
        """Tramos consecutivos de una misma fase: (fase, día de inicio, día de fin)."""
        tramos, ini = [], 0
        f = self.fase_por_dia
        for i in range(1, len(f) + 1):
            if i == len(f) or f[i] != f[ini]:
                tramos.append((f[ini], float(self.tiempos[ini]), float(self.tiempos[i - 1]) + 1))
                ini = i
        return tramos

    def fase_actual(self) -> str:
        """Fase del último día registrado."""
        return str(self.fase_por_dia[-1])

    def duracion_por_fase(self) -> dict[str, int]:
        return {f: int(np.sum(self.fase_por_dia == f)) for f in FASES}

    def es_coherente(self) -> bool:
        """True si la secuencia de fases respeta el orden del proceso.

        Transiciones válidas: Mesófila→Termófila→Enfriamiento→Maduración, más
        Enfriamiento→Termófila (recalentamiento, p. ej., tras un volteo).
        La serie debe comenzar en Mesófila.
        """
        orden = [s[0] for s in self.segmentos()]
        if not orden or orden[0] != "Mesófila":
            return False
        return all((a, b) in TRANSICIONES_VALIDAS for a, b in zip(orden, orden[1:]))


TRANSICIONES_VALIDAS = {
    ("Mesófila", "Termófila"),
    ("Termófila", "Enfriamiento"),
    ("Enfriamiento", "Maduración"),
    ("Enfriamiento", "Termófila"),   # recalentamiento
}

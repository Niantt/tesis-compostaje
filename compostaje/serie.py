"""Representación de una serie de temperatura de compostaje."""

from dataclasses import dataclass, field

import numpy as np

FASES = ["Mesófila", "Termófila", "Enfriamiento", "Maduración"]
COLORES = {"Mesófila": "#9ecae1", "Termófila": "#fc9272",
           "Enfriamiento": "#fdd0a2", "Maduración": "#a1d99b"}


@dataclass
class SerieTemperatura:
    """Una serie de temperatura.

    tiempos: días desde el inicio del registro (0 = primera medición).
    temperaturas: °C, misma longitud que tiempos.
    fase_ref: fase de referencia por punto (solo series etiquetadas).
    """
    nombre: str
    tiempos: np.ndarray
    temperaturas: np.ndarray
    fase_ref: np.ndarray | None = None
    metadatos: dict = field(default_factory=dict)

    def __post_init__(self):
        self.tiempos = np.asarray(self.tiempos, dtype=float)
        self.temperaturas = np.asarray(self.temperaturas, dtype=float)
        if self.fase_ref is not None:
            self.fase_ref = np.asarray(self.fase_ref, dtype=object)
        if len(self.tiempos) != len(self.temperaturas):
            raise ValueError("tiempos y temperaturas deben tener el mismo largo")

    def __len__(self):
        return len(self.tiempos)

    def duracion(self) -> float:
        """Duración en días entre la primera y la última medición."""
        return float(self.tiempos.max() - self.tiempos.min()) if len(self) else 0.0

    def es_periodica(self, periodo: float = 1.0, tol: float = 1e-6) -> bool:
        """True si las mediciones están separadas exactamente por `periodo` días."""
        if len(self) < 2:
            return True
        return bool(np.all(np.abs(np.diff(self.tiempos) - periodo) < tol))

    @property
    def etiquetada(self) -> bool:
        return self.fase_ref is not None and not all(f is None for f in self.fase_ref)

"""
entrenar_modelo.py — Tesis INFO-1197, Etapa 3
Construye el modelo de fases de DTW (ventanas etiquetadas de cada fase) con todas
las series etiquetadas del conjunto y lo guarda para la aplicación de escritorio.

Uso: python experimentos/entrenar_modelo.py [ruta.xlsx]
Salida: app/recursos/modelo_fases.npz
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from compostaje import ModeloFases  # noqa: E402
from compostaje.compilacion import cargar_conjunto  # noqa: E402

ruta = sys.argv[1] if len(sys.argv) > 1 else "Compilacion_temperatura_compost.xlsx"
series, _, _ = cargar_conjunto(ruta)
etiquetadas = [s for s in series if s.etiquetada]
modelo = ModeloFases(ventana=21).entrenar(etiquetadas)
destino = RAIZ / "app" / "recursos" / "modelo_fases.npz"
destino.parent.mkdir(parents=True, exist_ok=True)
modelo.guardar(destino)
print(f"Modelo con {len(modelo.y)} ventanas de {len(etiquetadas)} series -> {destino}")

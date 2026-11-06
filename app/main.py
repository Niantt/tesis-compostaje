"""Punto de entrada de la aplicación de escritorio.

Uso: python -m app.main   (desde la carpeta raíz del proyecto)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.controlador import ControladorPrincipal  # noqa: E402
from app.vista import VentanaPrincipal  # noqa: E402


def main():
    app = QApplication(sys.argv)
    controlador = ControladorPrincipal(VentanaPrincipal)
    controlador.vista.show()
    if len(sys.argv) > 1:                      # permite abrir un archivo desde la línea de comandos
        controlador.abrir_archivo(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

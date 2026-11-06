"""Capa Controlador: coordina la Vista y el Modelo (casos de uso CU-01 a CU-04)."""

import threading
import time
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

import numpy as np

from compostaje import (ErrorImportacion, SerieTemperatura, ExportadorReporte, IdentificadorDTW,
                        IdentificadorHibrido, IdentificadorPuntosCambio, ImportadorDatos,
                        ModeloFases, Preprocesador, ValidadorDatos)

RUTA_MODELO = Path(__file__).resolve().parent / "recursos" / "modelo_fases.npz"


class ControladorPrincipal:

    def __init__(self, vista_cls, ruta_modelo=RUTA_MODELO):
        self.importador = ImportadorDatos()
        self.validador = ValidadorDatos()
        self.preprocesador = Preprocesador()
        self.exportador = ExportadorReporte()
        self.tecnicas = {"Puntos de cambio": IdentificadorPuntosCambio()}
        if Path(ruta_modelo).exists():
            modelo = ModeloFases.cargar(ruta_modelo)
            # Configuración final de la Etapa 3: número de tramos por derivada discreta (D-13)
            self.tecnicas = {"Híbrido (recomendada)": IdentificadorHibrido(
                                 modelo, n_tramos="derivada", umbral_derivada=1.0),
                             "DTW": IdentificadorDTW(modelo), **self.tecnicas}
        self.serie_cruda = self.serie_actual = self.resultado = None

        self.vista = vista_cls(list(self.tecnicas))
        self.vista.abrir.connect(self.abrir_archivo)
        self.vista.identificar.connect(self.identificar_fases)
        self.vista.exportar.connect(self.exportar)
        if Path(ruta_modelo).exists():
            # La primera llamada a DTW compila código (unos 6 s). Se hace en segundo
            # plano al abrir la app para que el usuario no perciba la espera.
            threading.Thread(target=self._precalentar, daemon=True).start()
        else:
            self.vista.mostrar_estado("No se encontró el modelo de fases: solo está disponible "
                                      "la técnica de puntos de cambio.")

    def _precalentar(self):
        t = np.arange(30.0)
        serie = self.preprocesador.procesar(SerieTemperatura("_", t, 20 + 30 * np.exp(-((t - 8) / 6) ** 2)))
        self.tecnicas["DTW"].identificar(serie)

    # CU-01 importar + CU-02 visualizar
    def abrir_archivo(self, ruta):
        try:
            cruda = self.importador.leer(ruta)
        except ErrorImportacion as e:
            self.vista.mostrar_error(str(e))
            return
        avisos = self.validador.validar(cruda)
        if self.validador.hay_errores(avisos):
            self.vista.mostrar_error("\n".join(a.mensaje for a in avisos if a.nivel == "error"))
            return
        self.serie_cruda = cruda
        self.serie_actual = self.preprocesador.procesar(cruda)
        self.resultado = None
        self.vista.mostrar_serie(cruda, self.serie_actual, avisos)
        self.vista.habilitar_acciones(serie=True, resultado=False)
        self.vista.mostrar_estado(f"Registro cargado: {Path(ruta).name}")

    # CU-03 identificar fases
    def identificar_fases(self, tecnica):
        if self.serie_actual is None:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            t0 = time.perf_counter()
            self.resultado = self.tecnicas[tecnica].identificar(self.serie_actual)
            segundos = time.perf_counter() - t0
        except Exception as e:  # el usuario debe ver el problema, no un cierre de la app
            self.vista.mostrar_error(f"Error al identificar las fases: {e}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.vista.mostrar_resultado(self.resultado, segundos)
        self.vista.habilitar_acciones(serie=True, resultado=True)

    # CU-04 exportar (CSV; el reporte PDF corresponde a la Etapa 4)
    def exportar(self, ruta):
        try:
            self.exportador.exportar_csv(self.serie_actual, self.resultado, ruta)
            self.vista.mostrar_estado(f"Fases exportadas a {ruta}")
        except OSError as e:
            self.vista.mostrar_error(f"No se pudo guardar el archivo: {e}")

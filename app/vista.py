"""Capa Vista: ventana principal y gráfico de fases (PySide6 + pyqtgraph)."""

import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QColor, QFont
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHeaderView, QLabel, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox, QSplitter,
                               QTableWidget, QTableWidgetItem, QToolBar, QVBoxLayout, QWidget)

from compostaje import COLORES, FASES

pg.setConfigOptions(antialias=True, background="w", foreground="k")


class GraficoFases(pg.PlotWidget):
    """Curva de temperatura con las fases coloreadas de fondo (RF-03, RF-06)."""

    def __init__(self):
        super().__init__()
        self.setLabel("bottom", "Días desde el inicio del registro")
        self.setLabel("left", "Temperatura (°C)")
        self.showGrid(x=True, y=True, alpha=0.2)
        self.addLegend(offset=(-10, 10))
        self._regiones = []

    def mostrar_curva(self, cruda, procesada):
        self.clear()
        self._regiones = []
        self.addItem(pg.InfiniteLine(40, angle=0, pen=pg.mkPen("#888", style=Qt.DashLine),
                                     label="40 °C", labelOpts=dict(position=0.03, color="#666")))
        self.plot(cruda.tiempos, cruda.temperaturas, pen=None, symbol="o", symbolSize=4,
                  symbolBrush="#9a9a9a", symbolPen=None, name="Medición")
        self.plot(procesada.tiempos, procesada.temperaturas, pen=pg.mkPen("k", width=2),
                  name="Curva suavizada")
        self.enableAutoRange()

    def mostrar_fases(self, resultado):
        for r in self._regiones:
            self.removeItem(r)
        self._regiones = []
        for fase, ini, fin in resultado.segmentos():
            color = QColor(COLORES[fase])
            color.setAlpha(140)
            reg = pg.LinearRegionItem((ini, fin), movable=False, brush=color, pen=pg.mkPen(None))
            reg.setZValue(-10)
            self.addItem(reg)
            self._regiones.append(reg)


class VentanaPrincipal(QMainWindow):
    """Barra de acciones, gráfico central y panel lateral con el resumen (mockup de la Etapa 2)."""

    abrir = Signal(str)
    identificar = Signal(str)
    exportar = Signal(str)

    def __init__(self, tecnicas):
        super().__init__()
        self.setWindowTitle("Fases del compostaje")
        self.resize(1150, 650)

        barra = QToolBar("Acciones")
        barra.setMovable(False)
        self.addToolBar(barra)
        self.accion_abrir = QAction("Abrir registro…", self)
        self.accion_abrir.triggered.connect(self._elegir_archivo)
        barra.addAction(self.accion_abrir)
        barra.addSeparator()
        barra.addWidget(QLabel(" Técnica: "))
        self.combo_tecnica = QComboBox()
        self.combo_tecnica.addItems(tecnicas)
        barra.addWidget(self.combo_tecnica)
        self.accion_identificar = QAction("Identificar fases", self)
        self.accion_identificar.triggered.connect(
            lambda: self.identificar.emit(self.combo_tecnica.currentText()))
        barra.addAction(self.accion_identificar)
        barra.addSeparator()
        self.accion_exportar = QAction("Exportar CSV…", self)
        self.accion_exportar.triggered.connect(self._elegir_destino)
        barra.addAction(self.accion_exportar)

        self.grafico = GraficoFases()

        lateral = QWidget()
        capa = QVBoxLayout(lateral)
        self.lbl_serie = QLabel("Ningún registro abierto")
        self.lbl_serie.setWordWrap(True)
        self.lbl_serie.setFont(QFont("", 11, QFont.Bold))
        self.lbl_fase = QLabel("—")
        self.lbl_fase.setAlignment(Qt.AlignCenter)
        self.lbl_fase.setFont(QFont("", 15, QFont.Bold))
        self.lbl_fase.setMinimumHeight(48)
        self.tabla = QTableWidget(len(FASES), 2)
        self.tabla.setHorizontalHeaderLabels(["Fase", "Días"])
        self.tabla.verticalHeader().setVisible(False)
        self.tabla.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabla.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tabla.setMaximumHeight(150)
        for i, f in enumerate(FASES):
            item = QTableWidgetItem(f)
            item.setBackground(QColor(COLORES[f]))
            self.tabla.setItem(i, 0, item)
            self.tabla.setItem(i, 1, QTableWidgetItem("—"))
        self.lista_avisos = QListWidget()
        self.lista_avisos.setWordWrap(True)
        capa.addWidget(self.lbl_serie)
        capa.addWidget(QLabel("Fase actual"))
        capa.addWidget(self.lbl_fase)
        capa.addWidget(QLabel("Duración de cada fase"))
        capa.addWidget(self.tabla)
        capa.addWidget(QLabel("Avisos de validación"))
        capa.addWidget(self.lista_avisos)

        division = QSplitter()
        division.addWidget(self.grafico)
        division.addWidget(lateral)
        division.setSizes([820, 300])
        self.setCentralWidget(division)
        self.habilitar_acciones(serie=False, resultado=False)
        self.statusBar().showMessage("Abra un registro de temperatura (CSV o XLSX).")

    # --- interacción con el usuario ---------------------------------------
    def _elegir_archivo(self):
        ruta = self.seleccionar_archivo()
        if ruta:
            self.abrir.emit(ruta)

    def seleccionar_archivo(self) -> str:
        ruta, _ = QFileDialog.getOpenFileName(self, "Abrir registro de temperatura", "",
                                              "Registros (*.csv *.xlsx *.xls)")
        return ruta

    def _elegir_destino(self):
        ruta, _ = QFileDialog.getSaveFileName(self, "Exportar fases", "fases.csv", "CSV (*.csv)")
        if ruta:
            self.exportar.emit(ruta)

    # --- actualizaciones que llama el controlador --------------------------
    def habilitar_acciones(self, serie: bool, resultado: bool):
        self.accion_identificar.setEnabled(serie)
        self.combo_tecnica.setEnabled(serie)
        self.accion_exportar.setEnabled(resultado)

    def mostrar_serie(self, cruda, procesada, avisos):
        self.grafico.mostrar_curva(cruda, procesada)
        self.lbl_serie.setText(f"{cruda.nombre}\n{len(cruda)} mediciones · {procesada.duracion():.0f} días")
        self.lbl_fase.setText("—")
        self.lbl_fase.setStyleSheet("")
        for i in range(len(FASES)):
            self.tabla.item(i, 1).setText("—")
        self.lista_avisos.clear()
        for a in avisos:
            item = QListWidgetItem(("⛔ " if a.nivel == "error" else "⚠ ") + a.mensaje)
            self.lista_avisos.addItem(item)
        if not avisos:
            self.lista_avisos.addItem("✓ Sin observaciones")

    def mostrar_resultado(self, resultado, segundos):
        self.grafico.mostrar_fases(resultado)
        fase = resultado.fase_actual()
        self.lbl_fase.setText(fase)
        self.lbl_fase.setStyleSheet(f"background:{COLORES[fase]}; border-radius:6px;")
        for i, (f, n) in enumerate(resultado.duracion_por_fase().items()):
            self.tabla.item(i, 1).setText(str(n))
        coh = "secuencia coherente" if resultado.es_coherente() else "secuencia NO coherente"
        self.statusBar().showMessage(f"{resultado.tecnica}: {len(resultado.segmentos())} tramos, "
                                     f"{coh} · {segundos:.2f} s")

    def mostrar_error(self, mensaje: str):
        QMessageBox.warning(self, "No se pudo completar la acción", mensaje)

    def mostrar_estado(self, mensaje: str):
        self.statusBar().showMessage(mensaje)

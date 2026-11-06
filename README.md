# Identificación de fases del compostaje

Trabajo de Título INFO-1197 — Nicolás Escobar Videla
Profesor guía: Ricardo Soto Catalán · Profesor co-guía: Andrés Pérez-San Martín

Software de escritorio que identifica las fases del proceso de compostaje
(mesófila, termófila, enfriamiento y maduración) a partir de la curva de
temperatura de una pila.

Se implementan y contrastan tres técnicas de análisis de series temporales:

- **Detección de puntos de cambio**, que segmenta la curva en tramos con comportamiento homogéneo.
- **Dynamic Time Warping (DTW)**, que compara la forma de la curva con ventanas de referencia de cada fase.
- **Híbrido** (Etapa 3, técnica recomendada): los puntos de cambio proponen los tramos y DTW
  decide la fase de cada uno, respetando el orden del proceso y la regla térmica de 40 °C. El
  número de tramos se calcula en cada curva con la derivada discreta de la temperatura.

## Estado del proyecto

| Etapa | Contenido | Estado |
|---|---|---|
| 1 | Definición del problema, objetivos y requerimientos | Entregada |
| 2 | Diseño del software e implementación inicial (núcleo de análisis) | Entregada |
| 3 | Enfoque híbrido, aplicación de escritorio, pruebas y validación preliminar | Entregada |
| 4 | Validación final e informe | Pendiente |

## Estructura del repositorio

```
├── compostaje/            Núcleo de análisis (capa Modelo del diseño MVC)
│   ├── serie.py             SerieTemperatura
│   ├── importador.py        ImportadorDatos (CSV/XLSX)
│   ├── validador.py         ValidadorDatos
│   ├── preprocesador.py     Preprocesador (periodo diario, PCHIP y filtro de ruido)
│   ├── filtros.py           Savitzky-Golay, mediana + media móvil y polinomio de grado 12
│   ├── identificadores.py   IdentificadorFases y sus tres estrategias, ModeloFases
│   ├── knn_dtw.py           Plantillas por fase y clasificador KNN-DTW causal
│   ├── resultado.py         ResultadoFases
│   ├── exportador.py        ExportadorReporte
│   └── compilacion.py       Carga de la compilación de datos del proyecto
├── app/                   Aplicación de escritorio (Vista y Controlador)
│   ├── main.py
│   ├── vista.py             VentanaPrincipal, GraficoFases
│   └── controlador.py       ControladorPrincipal
├── experimentos/          Evaluación de técnicas, rendimiento y entrenamiento del modelo
├── tests/                 Pruebas unitarias, de integración y de regresión (pytest)
├── etapa2/                Scripts de la Etapa 2, sin cambios (línea base "antes")
└── informe/               Informe LaTeX de la etapa vigente
```

## Instalación

Python 3.10 o superior:

```
python -m pip install -r requirements.txt
```

## Datos

Los scripts usan `Compilacion_temperatura_compost.xlsx`, que **no se incluye en el
repositorio**. Cópialo en la carpeta raíz del proyecto.

## Uso

```
python experimentos/entrenar_modelo.py     # genera app/recursos/modelo_fases.npz (una vez)
python -m app.main                         # abre la aplicación
python -m pytest                           # pruebas automáticas
```

Sin el modelo de fases, la aplicación funciona solo con la técnica de puntos de cambio.
Acepta archivos CSV o XLSX con una columna de tiempo (fechas o días transcurridos) y una de
temperatura en °C.

### Resultados del informe

Cada script escribe sus tablas y figuras en `salida_etapa3/`:

| Script | Resultado | Tiempo aprox. |
|---|---|---|
| `experimentos/evaluar_tecnicas.py` | Comparación antes/después, coherencia y número de tramos | 3 min |
| `experimentos/comparar_filtros.py` | Comparación de filtros de ruido | 15 min |
| `experimentos/plantillas_adaptativas.py` | Subcurvas cortas frente a la ventana de 21 días | 5 min |
| `experimentos/plantillas_modelo.py` | Plantilla modelo por fase y comparación con las 5 series | 1 min |
| `experimentos/bootstrap_oob.py` | Validación bootstrap OOB (1.000 repeticiones) | 25 min |
| `experimentos/validacion_campo.py` | Validación con anotaciones de campo (pilas 15 y 11) | 1 min |
| `experimentos/rendimiento.py` | Pruebas de rendimiento y estrés | 3 min |
| `experimentos/material_etiquetado.py` | PDF y planilla para que el co-guía etiquete las pilas | 1 min |

Para reproducir los resultados de la Etapa 2 (la prueba de regresión los compara con el paquete):

```
cd etapa2
python preprocesamiento.py ../Compilacion_temperatura_compost.xlsx
python changepoint.py
python dtw_fases.py
```

## Informe

El informe se compila desde `informe/main.tex`, con `pdflatex` o en Overleaf.
Los diagramas se editan desde `informe/figuras/diagramas/fuentes/`.

# Identificación de fases del compostaje mediante series temporales de temperatura

Trabajo de Título (INFO-1197) — Universidad Católica de Temuco

**Estudiante:** Nicolás Escobar Videla
**Profesor guía:** Ricardo Soto Catalán
**Profesor co-guía:** Andrés Pérez-San Martín

## Descripción

Software para identificar automáticamente la fase del proceso de compostaje
(mesófila, termófila, enfriamiento o maduración) a partir de los registros de
temperatura de la pila. Se implementan y contrastan dos técnicas de análisis de
series temporales:

- **Detección de puntos de cambio**, que segmenta la curva en tramos con comportamiento homogéneo.
- **Dynamic Time Warping (DTW)**, que compara la forma de la curva con modelos de referencia de cada fase.

- ## Estado del proyecto

| Etapa | Contenido | Estado |
|---|---|---|
| 1 | Definición del problema, objetivos y requerimientos | Entregada |
| 2 | Diseño del software e implementación inicial (núcleo de análisis) | En proceso |
| 3 | Integración con la interfaz gráfica, pruebas y validación preliminar | Pendiente |
| 4 | Validación final e informe | Pendiente |

## Requisitos

- Python 3.10 o superior
- Las dependencias de `requirements.txt`

```
python -m pip install -r requirements.txt
```

## Datos

Los scripts utilizan la compilación de registros de temperatura de compostaje
`Compilacion_temperatura_compost.xlsx`, que **no se incluye en el repositorio**.
Para ejecutar los scripts, copia el archivo en la carpeta raíz del proyecto.


## Uso

Los scripts se ejecutan en orden, ya que cada uno utiliza los resultados del anterior:

```
python preprocesamiento.py   # genera salida/
python changepoint.py        # genera salida_changepoint/
python dtw_fases.py          # genera salida_dtw/
```

| Carpeta | Contenido |
|---|---|
| `salida/` | Inventario de series, criterios de selección, series preprocesadas, tablas LaTeX y figuras |
| `salida_changepoint/` | Puntos de cambio detectados, evaluación y figuras |
| `salida_dtw/` | Fases asignadas por DTW, evaluación, comparación de técnicas y figuras |

Las carpetas de resultados no se suben al repositorio, porque se recrean al ejecutar los scripts.

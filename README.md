# Red Vial en Crisis - Proyecto de Modelación y Simulación
## CC3039 - Universidad del Valle de Guatemala

**Autores:**
- Michelle Angel de María Mejía Villela, 22596
- Silvia Alejandra Illescas Fernandez, 22376

---

## Descripción del Proyecto

Este proyecto analiza la resiliencia de la red vial urbana de Guatemala frente a incidentes locales (accidentes, cierres de calles) mediante modelación y simulación basada en agentes.

### Objetivos

**General:** Modelar y simular la red vial urbana de Guatemala para evaluar su resiliencia frente a incidentes locales.

**Específicos:**
- Construir el grafo de la red vial a partir de datos de OpenStreetMap
- Implementar un modelo de simulación de agentes con rutas OD (Origen-Destino)
- Evaluar el impacto de incidentes en tiempos de viaje y congestión
- Analizar la vulnerabilidad de la red en función de su topología
- Presentar recomendaciones para mejorar la resiliencia del sistema vial

---

## Estructura del Proyecto

    red-vial-guatemala/
    │
    ├── data/                          # Datos del proyecto
    │   ├── raw/                       # Datos crudos de OSM
    │   ├── processed/                 # Grafos procesados
    │   └── results/                   # Resultados de simulaciones
    │
    ├── src/                           # Código fuente
    │   ├── __init__.py
    │   ├── data_acquisition.py        # Obtención de datos de OSM
    │   ├── network_model.py           # Modelado del grafo vial
    │   ├── agent_simulation.py        # Simulación de agentes vehiculares
    │   ├── incident_manager.py        # Gestión de incidentes
    │   ├── metrics_analyzer.py        # Cálculo de métricas
    │   └── visualization.py           # Generación de visualizaciones
    │
    ├── notebooks/                     # Jupyter notebooks para análisis
    │   ├── 01_exploracion_datos.ipynb
    │   ├── 02_construccion_red.ipynb
    │   ├── 03_simulacion_baseline.ipynb
    │   ├── 04_analisis_incidentes.ipynb
    │   └── 05_resultados_finales.ipynb
    │
    ├── scripts/                       # Scripts ejecutables
    │   ├── run_simulation.py          # Ejecutar simulación completa
    │   └── generate_report.py         # Generar informe automático
    │
    ├── tests/                         # Pruebas unitarias
    │   └── test_network.py
    │
    ├── docs/                          # Documentación
    │   ├── informe.md                 # Informe
    │   └── presentacion.md            # Guía para presentación
    │
    ├── results/                       # Visualizaciones y reportes
    │   ├── figures/
    │   └── videos/
    │
    ├── requirements.txt               # Dependencias
    ├── config.yaml                    # Configuración del proyecto
    └── README.md                      # Este archivo

# Crear ambiente virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt


## Métricas a Evaluar

1. **Tiempo medio de viaje**: Promedio de tiempo que toman los agentes en completar sus rutas
2. **Retraso acumulado**: Tiempo adicional debido a incidentes
3. **Carga en aristas críticas**: Nivel de congestión en rutas principales
4. **Porcentaje de re-enrutamientos**: Vehículos que cambian de ruta
5. **Resiliencia de la red**: Capacidad de mantener flujo bajo incidentes

---

##  Visualizaciones

- Mapa interactivo de la red vial (Folium)
- Histogramas de duraciones de viaje
- Series temporales de vehículos en el sistema
- Mapas de calor de congestión
- Grafos de la red con nodos críticos resaltados
- Comparación baseline vs. incidentes

---

## Tecnologías Utilizadas

- **Python 3.8+**
- **OSMnx**: Obtención de datos de OpenStreetMap
- **NetworkX**: Análisis de grafos
- **Pandas**: Manipulación de datos
- **Matplotlib/Seaborn**: Visualizaciones estáticas
- **Folium**: Mapas interactivos
- **GeoPandas**: Datos geoespaciales
- **NumPy/SciPy**: Cálculos numéricos

---


## Referencias

- Ross, S. (2013). *Simulation*. Academic Press.
- OpenStreetMap Contributors. (2025). OpenStreetMap.
- Boeing, G. (2017). OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks.

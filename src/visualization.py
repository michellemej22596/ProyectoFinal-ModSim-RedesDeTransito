"""
Módulo para la visualización de resultados.

Este módulo proporciona funciones para crear mapas, gráficas y
visualizaciones interactivas de la red y resultados de simulación.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import yaml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar estilo de matplotlib
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10


class Visualizer:
    """
    Clase para crear visualizaciones de la red y simulación.
    """
    
    def __init__(self, G: nx.MultiDiGraph, config_path: str = "config.yaml"):
        """
        Inicializa el visualizador.
        
        Args:
            G: Grafo de la red vial
            config_path: Ruta al archivo de configuración
        """
        self.G = G
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.viz_config = self.config['visualization']
        self.figures_path = Path(self.config['paths']['figures'])
        self.figures_path.mkdir(parents=True, exist_ok=True)
        
        # Colores para consistencia
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#06A77D',
            'warning': '#F18F01',
            'danger': '#C73E1D',
            'info': '#6A7FDB'
        }
    
    def plot_network_map(
        self,
        critical_nodes: Optional[List] = None,
        critical_edges: Optional[List] = None,
        save_filename: str = "network_map.html"
    ) -> folium.Map:
        """
        Crea un mapa interactivo de la red vial.
        
        Args:
            critical_nodes: Lista de nodos críticos a resaltar
            critical_edges: Lista de aristas críticas a resaltar
            save_filename: Nombre del archivo para guardar
            
        Returns:
            Objeto folium.Map
        """
        logger.info("Creando mapa de la red...")
        
        # Obtener centro del mapa
        center = self.config['location']['center_point']
        
        # Crear mapa base
        m = folium.Map(
            location=center,
            zoom_start=12,
            tiles=self.viz_config['map_tiles']
        )
        
        # Agregar todas las aristas
        for u, v, data in self.G.edges(data=True):
            # Obtener coordenadas
            u_coords = (self.G.nodes[u]['y'], self.G.nodes[u]['x'])
            v_coords = (self.G.nodes[v]['y'], self.G.nodes[v]['x'])
            
            # Determinar color y peso
            is_critical = False
            if critical_edges:
                is_critical = any(
                    (u == edge[0] and v == edge[1]) or (v == edge[0] and u == edge[1])
                    for edge in critical_edges
                )
            
            color = self.colors['danger'] if is_critical else self.colors['primary']
            weight = 4 if is_critical else 2
            opacity = 0.8 if is_critical else 0.4
            
            # Crear línea
            folium.PolyLine(
                locations=[u_coords, v_coords],
                color=color,
                weight=weight,
                opacity=opacity,
                popup=f"Vía: {data.get('name', 'Sin nombre')}<br>"
                      f"Tipo: {data.get('highway', 'N/A')}<br>"
                      f"Longitud: {data.get('length', 0)/1000:.2f} km"
            ).add_to(m)
        
        # Agregar nodos críticos
        if critical_nodes:
            for node in critical_nodes:
                coords = (self.G.nodes[node]['y'], self.G.nodes[node]['x'])
                
                folium.CircleMarker(
                    location=coords,
                    radius=8,
                    color=self.colors['warning'],
                    fill=True,
                    fillColor=self.colors['warning'],
                    fillOpacity=0.7,
                    popup=f"Nodo crítico: {node}"
                ).add_to(m)
        
        # Agregar leyenda
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 200px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><strong>Leyenda</strong></p>
        <p><span style="color:''' + self.colors['primary'] + ''';">━━━</span> Vías normales</p>
        <p><span style="color:''' + self.colors['danger'] + ''';">━━━</span> Vías críticas</p>
        <p><span style="color:''' + self.colors['warning'] + ''';">●</span> Nodos críticos</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Guardar
        filepath = self.figures_path / save_filename
        m.save(str(filepath))
        logger.info(f"Mapa guardado en {filepath}")
        
        return m
    
    def plot_travel_time_distribution(
        self,
        scenarios_results: Dict[str, Dict],
        save_filename: str = "travel_time_distribution.png"
    ):
        """
        Crea histogramas de distribución de tiempos de viaje.
        
        Args:
            scenarios_results: Diccionario {nombre_escenario: resultados}
            save_filename: Nombre del archivo para guardar
        """
        logger.info("Creando gráfica de distribución de tiempos de viaje...")
        
        fig, axes = plt.subplots(
            len(scenarios_results), 1,
            figsize=(12, 4 * len(scenarios_results)),
            squeeze=False
        )
        
        for idx, (scenario_name, results) in enumerate(scenarios_results.items()):
            ax = axes[idx, 0]
            
            travel_times = np.array(results['travel_times']) / 60  # Convertir a minutos
            
            # Histograma
            ax.hist(
                travel_times,
                bins=50,
                color=self.colors['primary'],
                alpha=0.7,
                edgecolor='black'
            )
            
            # Líneas de estadísticas
            mean_time = np.mean(travel_times)
            median_time = np.median(travel_times)
            
            ax.axvline(mean_time, color=self.colors['danger'], 
                      linestyle='--', linewidth=2, label=f'Media: {mean_time:.1f} min')
            ax.axvline(median_time, color=self.colors['success'], 
                      linestyle='--', linewidth=2, label=f'Mediana: {median_time:.1f} min')
            
            ax.set_xlabel('Tiempo de Viaje (minutos)')
            ax.set_ylabel('Frecuencia')
            ax.set_title(f'Distribución de Tiempos de Viaje - {scenario_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = self.figures_path / save_filename
        plt.savefig(filepath, dpi=self.viz_config['dpi'], bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráfica guardada en {filepath}")
    
    def plot_vehicles_over_time(
        self,
        scenarios_results: Dict[str, Dict],
        save_filename: str = "vehicles_over_time.png"
    ):
        """
        Crea gráfica de vehículos en el sistema a lo largo del tiempo.
        
        Args:
            scenarios_results: Diccionario {nombre_escenario: resultados}
            save_filename: Nombre del archivo para guardar
        """
        logger.info("Creando gráfica de vehículos en el tiempo...")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        colors_list = list(self.colors.values())
        
        for idx, (scenario_name, results) in enumerate(scenarios_results.items()):
            if 'history' in results:
                history_df = results['history']
                
                ax.plot(
                    history_df['time'] / 60,  # Convertir a minutos
                    history_df['active_vehicles'],
                    label=scenario_name,
                    color=colors_list[idx % len(colors_list)],
                    linewidth=2
                )
        
        ax.set_xlabel('Tiempo (minutos)')
        ax.set_ylabel('Vehículos Activos en el Sistema')
        ax.set_title('Evolución de Vehículos Activos Durante la Simulación')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = self.figures_path / save_filename
        plt.savefig(filepath, dpi=self.viz_config['dpi'], bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráfica guardada en {filepath}")
    
    def plot_scenario_comparison(
        self,
        comparison_df: pd.DataFrame,
        save_filename: str = "scenario_comparison.png"
    ):
        """
        Crea gráfica de barras comparando escenarios.
        
        Args:
            comparison_df: DataFrame con comparación de escenarios
            save_filename: Nombre del archivo para guardar
        """
        logger.info("Creando gráfica de comparación de escenarios...")
        
        # Métricas a visualizar
        metrics = [
            'avg_travel_time_min',
            'completion_rate',
            'avg_reroutes',
            'network_efficiency'
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            if metric in comparison_df.columns:
                bars = ax.bar(
                    comparison_df['scenario'],
                    comparison_df[metric],
                    color=[self.colors['primary'], self.colors['danger'], 
                           self.colors['warning']][:len(comparison_df)]
                )
                
                # Agregar valores en las barras
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom'
                    )
                
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(metric.replace('_', ' ').title())
                ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        filepath = self.figures_path / save_filename
        plt.savefig(filepath, dpi=self.viz_config['dpi'], bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráfica guardada en {filepath}")
    
    def create_heatmap(
        self,
        edge_loads: Dict,
        save_filename: str = "congestion_heatmap.html"
    ) -> folium.Map:
        """
        Crea mapa de calor de congestión.
        
        Args:
            edge_loads: Diccionario con carga de aristas
            save_filename: Nombre del archivo para guardar
            
        Returns:
            Objeto folium.Map
        """
        logger.info("Creando mapa de calor de congestión...")
        
        center = self.config['location']['center_point']
        m = folium.Map(location=center, zoom_start=12)
        
        # Normalizar cargas
        max_load = max(edge_loads.values()) if edge_loads else 1
        
        for (u, v, k), load in edge_loads.items():
            try:
                u_coords = (self.G.nodes[u]['y'], self.G.nodes[u]['x'])
                v_coords = (self.G.nodes[v]['y'], self.G.nodes[v]['x'])
                
                # Color basado en carga
                intensity = load / max_load if max_load > 0 else 0
                
                # Gradiente de verde a rojo
                if intensity < 0.33:
                    color = self.colors['success']
                elif intensity < 0.66:
                    color = self.colors['warning']
                else:
                    color = self.colors['danger']
                
                weight = 2 + intensity * 6
                
                folium.PolyLine(
                    locations=[u_coords, v_coords],
                    color=color,
                    weight=weight,
                    opacity=0.7,
                    popup=f"Carga: {load} vehículos"
                ).add_to(m)
                
            except KeyError:
                continue
        
        filepath = self.figures_path / save_filename
        m.save(str(filepath))
        logger.info(f"Mapa de calor guardado en {filepath}")
        
        return m


def main():
    """
    Función principal de demostración.
    """
    from src.data_acquisition import OSMDataAcquisition
    
    # Cargar la red
    acquisitor = OSMDataAcquisition()
    G = acquisitor.load_network()
    
    # Crear visualizador
    viz = Visualizer(G)
    
    # Crear mapa de la red
    viz.plot_network_map()
    
    logger.info("Visualizaciones de demostración creadas")


if __name__ == "__main__":
    main()

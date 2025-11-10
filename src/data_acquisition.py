"""
Módulo para la adquisición de datos de OpenStreetMap.

Este módulo maneja la descarga y preprocesamiento inicial de la red vial
desde OpenStreetMap utilizando OSMnx.
"""

import osmnx as ox
import networkx as nx
import pandas as pd
import geopandas as gpd
from pathlib import Path
import yaml
import pickle
from typing import Tuple, Optional
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OSMDataAcquisition:
    """
    Clase para manejar la adquisición de datos de OpenStreetMap.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa el adquisidor de datos.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.location = self.config['location']
        self.network_config = self.config['network']
        
        # Crear directorios si no existen
        Path(self.config['paths']['data_raw']).mkdir(parents=True, exist_ok=True)
        Path(self.config['paths']['data_processed']).mkdir(parents=True, exist_ok=True)
    
    def download_network(self, save: bool = True) -> Tuple[nx.MultiDiGraph, gpd.GeoDataFrame]:
        """
        Descarga la red vial desde OpenStreetMap.
        
        Args:
            save: Si True, guarda la red descargada
            
        Returns:
            Tupla con el grafo de la red y los nodos como GeoDataFrame
        """
        logger.info(f"Descargando red vial de {self.location['city']}...")
        
        try:
            # Descargar la red usando el punto central y radio
            G = ox.graph_from_point(
                center_point=self.location['center_point'],
                dist=self.location['radius_km'] * 1000,  # Convertir a metros
                network_type=self.location['network_type'],
                simplify=self.network_config['simplify'],
                retain_all=self.network_config['retain_all'],
                truncate_by_edge=self.network_config['truncate_by_edge']
            )
            
            logger.info(f"Red descargada: {len(G.nodes)} nodos, {len(G.edges)} aristas")
            
            # Agregar atributos de velocidad si no existen
            G = self._add_speed_attributes(G)
            
            # Calcular tiempos de viaje
            G = ox.add_edge_travel_times(G)
            
            if save:
                self.save_network(G)
            
            # Convertir nodos a GeoDataFrame para análisis
            nodes_gdf = ox.graph_to_gdfs(G, edges=False)
            
            return G, nodes_gdf
            
        except Exception as e:
            logger.error(f"Error al descargar la red: {e}")
            raise
    
    def _add_speed_attributes(self, G: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """
        Agrega atributos de velocidad a las aristas basados en el tipo de vía.
        
        Args:
            G: Grafo de la red
            
        Returns:
            Grafo con atributos de velocidad
        """
        speeds = self.config['simulation']['speeds']
        
        for u, v, key, data in G.edges(keys=True, data=True):
            # Asegurar que existe el atributo 'length'
            if 'length' not in data:
                # Calcular longitud si no existe
                try:
                    point_u = (G.nodes[u]['y'], G.nodes[u]['x'])
                    point_v = (G.nodes[v]['y'], G.nodes[v]['x'])
                    data['length'] = ox.distance.great_circle(point_u[0], point_u[1], point_v[0], point_v[1])
                except:
                    data['length'] = 100  # Valor por defecto en metros
            
            speed = None
            if 'maxspeed' in data and data['maxspeed'] is not None:
                maxspeed_value = data['maxspeed']
                
                # Si es una lista, tomar el primer valor
                if isinstance(maxspeed_value, list):
                    maxspeed_value = maxspeed_value[0] if maxspeed_value else None
                
                # Convertir maxspeed a numérico
                if maxspeed_value is not None:
                    if isinstance(maxspeed_value, str):
                        try:
                            # Manejar casos como "50 mph" o "50"
                            speed_str = maxspeed_value.split()[0]
                            speed = float(speed_str)
                        except:
                            speed = None
                    elif isinstance(maxspeed_value, (int, float)):
                        speed = float(maxspeed_value)
            
            # Si no se pudo obtener velocidad de maxspeed, usar tipo de vía
            if speed is None:
                highway_type = data.get('highway', 'default')
                
                if isinstance(highway_type, list):
                    highway_type = highway_type[0]
                
                # Mapear tipo de vía a velocidad
                speed = speeds.get(highway_type, speeds['default'])
            
            # Establecer speed_kph (requerido por OSMnx)
            data['speed_kph'] = speed
        
        return G
    
    def save_network(self, G: nx.MultiDiGraph, filename: str = "network_graph.gpickle"):
        """
        Guarda el grafo de la red.
        
        Args:
            G: Grafo a guardar
            filename: Nombre del archivo
        """
        filepath = Path(self.config['paths']['data_raw']) / filename
        
        # Guardar como pickle de NetworkX
        with open(filepath, 'wb') as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Red guardada en {filepath}")
        
        # También guardar como GraphML para compatibilidad
        graphml_path = filepath.with_suffix('.graphml')
        ox.save_graphml(G, graphml_path)
        logger.info(f"Red guardada en formato GraphML: {graphml_path}")
    
    def load_network(self, filename: str = "network_graph.gpickle") -> nx.MultiDiGraph:
        """
        Carga un grafo previamente guardado.
        
        Args:
            filename: Nombre del archivo
            
        Returns:
            Grafo de la red
        """
        filepath = Path(self.config['paths']['data_raw']) / filename
        
        if not filepath.exists():
            logger.warning(f"Archivo {filepath} no existe. Descargando red...")
            G, _ = self.download_network()
            return G
        
        with open(filepath, 'rb') as f:
            G = pickle.load(f)
        
        logger.info(f"Red cargada desde {filepath}")
        return G
    
    def get_network_summary(self, G: nx.MultiDiGraph) -> dict:
        """
        Genera un resumen estadístico de la red.
        
        Args:
            G: Grafo de la red
            
        Returns:
            Diccionario con estadísticas de la red
        """
        stats = ox.basic_stats(G)
        
        summary = {
            'num_nodes': len(G.nodes),
            'num_edges': len(G.edges),
            'average_degree': sum(dict(G.degree()).values()) / len(G.nodes),
            'network_density': nx.density(G),
            'total_length_km': sum(data.get('length', 0) for u, v, data in G.edges(data=True)) / 1000,
        }
        
        # Agregar estadísticas de OSMnx
        summary.update(stats)
        
        return summary


def main():
    """
    Función principal para ejecutar la adquisición de datos.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Descargar red vial de OpenStreetMap')
    parser.add_argument('--city', type=str, help='Nombre de la ciudad')
    parser.add_argument('--network-type', type=str, choices=['drive', 'walk', 'bike', 'all'],
                       help='Tipo de red a descargar')
    
    args = parser.parse_args()
    
    # Crear instancia del adquisidor
    acquisitor = OSMDataAcquisition()
    
    # Actualizar configuración si se proporcionaron argumentos
    if args.city:
        acquisitor.location['city'] = args.city
    if args.network_type:
        acquisitor.location['network_type'] = args.network_type
    
    # Descargar la red
    G, nodes_gdf = acquisitor.download_network()
    
    # Mostrar resumen
    summary = acquisitor.get_network_summary(G)
    
    print("\n" + "="*50)
    print("RESUMEN DE LA RED VIAL")
    print("="*50)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()

"""
Módulo para el modelado y análisis de la red vial como grafo.

Este módulo contiene funciones para analizar propiedades topológicas,
calcular métricas de centralidad, e identificar nodos y aristas críticas.
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import yaml
from pathlib import Path
import pickle
import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkModel:
    """
    Clase para modelar y analizar la red vial.
    """
    
    def __init__(self, G: nx.MultiDiGraph, config_path: str = "config.yaml"):
        """
        Inicializa el modelo de red.
        
        Args:
            G: Grafo de la red vial
            config_path: Ruta al archivo de configuración
        """
        self.G = G
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.metrics = {}
        
    def calculate_centrality_measures(self) -> Dict[str, Dict]:
        """
        Calcula medidas de centralidad para nodos.
        
        Returns:
            Diccionario con diferentes medidas de centralidad
        """
        logger.info("Calculando medidas de centralidad...")
        
        centrality = {}
        
        # Centralidad de grado
        centrality['degree'] = dict(self.G.degree())
        
        # Centralidad de cercanía (closeness)
        try:
            centrality['closeness'] = nx.closeness_centrality(self.G, distance='length')
        except:
            logger.warning("No se pudo calcular closeness centrality")
            centrality['closeness'] = {}
        
        # Centralidad de intermediación (betweenness)
        try:
            # Usar una muestra para grafos grandes
            if len(self.G.nodes) > 1000:
                k = min(100, len(self.G.nodes))
                centrality['betweenness'] = nx.betweenness_centrality(
                    self.G, k=k, weight='length'
                )
            else:
                centrality['betweenness'] = nx.betweenness_centrality(
                    self.G, weight='length'
                )
        except:
            logger.warning("No se pudo calcular betweenness centrality")
            centrality['betweenness'] = {}
        
        self.metrics['centrality'] = centrality
        
        logger.info("Medidas de centralidad calculadas")
        return centrality
    
    def identify_critical_nodes(self, top_n: int = 20) -> pd.DataFrame:
        """
        Identifica los nodos más críticos basados en centralidad.
        
        Args:
            top_n: Número de nodos críticos a retornar
            
        Returns:
            DataFrame con los nodos críticos y sus métricas
        """
        if 'centrality' not in self.metrics:
            self.calculate_centrality_measures()
        
        # Crear DataFrame con todas las medidas
        nodes_data = []
        
        for node in self.G.nodes():
            node_data = {
                'node_id': node,
                'degree': self.metrics['centrality']['degree'].get(node, 0),
                'closeness': self.metrics['centrality']['closeness'].get(node, 0),
                'betweenness': self.metrics['centrality']['betweenness'].get(node, 0),
            }
            
            # Agregar coordenadas
            node_attrs = self.G.nodes[node]
            node_data['lat'] = node_attrs.get('y', None)
            node_data['lon'] = node_attrs.get('x', None)
            
            nodes_data.append(node_data)
        
        df = pd.DataFrame(nodes_data)
        
        # Normalizar métricas para crear un score combinado
        for col in ['degree', 'closeness', 'betweenness']:
            if df[col].max() > 0:
                df[f'{col}_norm'] = df[col] / df[col].max()
            else:
                df[f'{col}_norm'] = 0
        
        df['criticality_score'] = (
            df['degree_norm'] * 0.3 +
            df['closeness_norm'] * 0.3 +
            df['betweenness_norm'] * 0.4
        )
        
        # Ordenar por criticality score
        critical_nodes = df.nlargest(top_n, 'criticality_score')
        
        return critical_nodes
    
    def identify_critical_edges(self, top_n: int = 20) -> pd.DataFrame:
        """
        Identifica las aristas más críticas (vías principales).
        
        Args:
            top_n: Número de aristas críticas a retornar
            
        Returns:
            DataFrame con las aristas críticas
        """
        logger.info("Identificando aristas críticas...")
        
        # Calcular betweenness para aristas
        try:
            edge_betweenness = nx.edge_betweenness_centrality(
                self.G, weight='length', normalized=True
            )
        except:
            logger.warning("Usando muestra para edge betweenness")
            k = min(100, len(self.G.nodes))
            edge_betweenness = nx.edge_betweenness_centrality(
                self.G, k=k, weight='length', normalized=True
            )
        
        # Crear DataFrame
        edges_data = []
        
        for (u, v, k), betweenness in edge_betweenness.items():
            edge_data = self.G[u][v][k]
            
            edges_data.append({
                'from_node': u,
                'to_node': v,
                'key': k,
                'betweenness': betweenness,
                'length': edge_data.get('length', 0),
                'highway': edge_data.get('highway', 'unknown'),
                'name': edge_data.get('name', 'Unnamed'),
                'maxspeed': edge_data.get('maxspeed', 40)
            })
        
        df = pd.DataFrame(edges_data)
        
        # Calcular score de criticidad
        df['length_km'] = df['length'] / 1000
        df['criticality_score'] = df['betweenness'] * df['length_km']
        
        critical_edges = df.nlargest(top_n, 'criticality_score')
        
        return critical_edges
    
    def analyze_network_topology(self) -> Dict:
        """
        Analiza propiedades topológicas de la red.
        
        Returns:
            Diccionario con propiedades topológicas
        """
        logger.info("Analizando topología de la red...")
        
        topology = {}
        
        # Conectividad
        if nx.is_strongly_connected(self.G):
            topology['is_connected'] = True
            topology['num_components'] = 1
        else:
            topology['is_connected'] = False
            topology['num_components'] = nx.number_strongly_connected_components(self.G)
        
        # Componente gigante
        largest_cc = max(nx.strongly_connected_components(self.G), key=len)
        topology['largest_component_size'] = len(largest_cc)
        topology['largest_component_pct'] = len(largest_cc) / len(self.G.nodes) * 100
        
        # Grado promedio
        degrees = [d for n, d in self.G.degree()]
        topology['average_degree'] = np.mean(degrees)
        topology['std_degree'] = np.std(degrees)
        
        # Coeficiente de clustering
        try:
            # Convertir a grafo no dirigido para clustering
            G_undirected = self.G.to_undirected()
            topology['average_clustering'] = nx.average_clustering(G_undirected)
        except:
            topology['average_clustering'] = None
        
        # Distribución de grados (para análisis de red libre de escala)
        degree_sequence = sorted(degrees, reverse=True)
        topology['max_degree'] = max(degrees)
        topology['min_degree'] = min(degrees)
        
        self.metrics['topology'] = topology
        
        return topology
    
    def calculate_shortest_paths_sample(self, sample_size: int = 100) -> pd.DataFrame:
        """
        Calcula una muestra de caminos más cortos para análisis.
        
        Args:
            sample_size: Número de pares origen-destino a muestrear
            
        Returns:
            DataFrame con información de los caminos
        """
        logger.info(f"Calculando {sample_size} caminos más cortos de muestra...")
        
        # Tomar muestra de nodos
        nodes = list(self.G.nodes())
        if len(nodes) > sample_size:
            sample_nodes = np.random.choice(nodes, size=sample_size, replace=False)
        else:
            sample_nodes = nodes
        
        paths_data = []
        
        for i, origin in enumerate(sample_nodes):
            # Seleccionar un destino aleatorio
            dest_nodes = [n for n in sample_nodes if n != origin]
            if not dest_nodes:
                continue
            
            destination = np.random.choice(dest_nodes)
            
            try:
                # Calcular camino más corto por tiempo
                path = nx.shortest_path(
                    self.G, origin, destination, weight='travel_time'
                )
                
                # Calcular métricas del camino
                path_length = sum(
                    self.G[path[j]][path[j+1]][0].get('length', 0)
                    for j in range(len(path)-1)
                )
                
                path_time = sum(
                    self.G[path[j]][path[j+1]][0].get('travel_time', 0)
                    for j in range(len(path)-1)
                )
                
                paths_data.append({
                    'origin': origin,
                    'destination': destination,
                    'path_nodes': len(path),
                    'path_length_km': path_length / 1000,
                    'path_time_min': path_time / 60,
                    'path': path
                })
                
            except nx.NetworkXNoPath:
                logger.warning(f"No hay camino entre {origin} y {destination}")
                continue
        
        df = pd.DataFrame(paths_data)
        
        return df
    
    def save_model(self, filename: str = "network_model.pkl"):
        """
        Guarda el modelo y sus métricas.
        
        Args:
            filename: Nombre del archivo
        """
        filepath = Path(self.config['paths']['data_processed']) / filename
        
        model_data = {
            'metrics': self.metrics,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Modelo guardado en {filepath}")


def main():
    """
    Función principal para ejecutar el análisis de red.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Analizar red vial')
    parser.add_argument('--build', action='store_true', help='Construir y analizar la red')
    args = parser.parse_args()
    
    if not args.build:
        print("Usa --build para construir y analizar la red")
        return
    
    from data_acquisition import OSMDataAcquisition
    
    # Cargar la red
    acquisitor = OSMDataAcquisition()
    G = acquisitor.load_network()
    
    # Crear modelo
    model = NetworkModel(G)
    
    # Análisis topológico
    print("\n" + "="*50)
    print("ANÁLISIS TOPOLÓGICO")
    print("="*50)
    topology = model.analyze_network_topology()
    for key, value in topology.items():
        print(f"{key}: {value}")
    
    # Identificar nodos críticos
    print("\n" + "="*50)
    print("NODOS CRÍTICOS")
    print("="*50)
    critical_nodes = model.identify_critical_nodes(top_n=10)
    print(critical_nodes[['node_id', 'criticality_score', 'degree', 'betweenness']])
    
    # Identificar aristas críticas
    print("\n" + "="*50)
    print("ARISTAS CRÍTICAS (VÍAS PRINCIPALES)")
    print("="*50)
    critical_edges = model.identify_critical_edges(top_n=10)
    print(critical_edges[['name', 'highway', 'criticality_score', 'length_km']])
    
    # Guardar modelo
    model.save_model()


if __name__ == "__main__":
    main()

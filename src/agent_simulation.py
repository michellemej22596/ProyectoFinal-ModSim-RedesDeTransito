"""
Módulo para la simulación basada en agentes del tráfico vehicular.

Este módulo implementa la lógica de simulación de vehículos como agentes
que se mueven a través de la red vial.
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import yaml
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Estados posibles de un agente."""
    WAITING = "waiting"
    TRAVELING = "traveling"
    ARRIVED = "arrived"
    REROUTING = "rerouting"


@dataclass
class Vehicle:
    """
    Clase que representa un vehículo (agente) en la simulación.
    """
    id: int
    origin: int
    destination: int
    current_node: int
    state: AgentState = AgentState.WAITING
    path: List[int] = field(default_factory=list)
    path_index: int = 0
    departure_time: float = 0.0
    arrival_time: Optional[float] = None
    total_distance: float = 0.0
    total_time: float = 0.0
    reroute_count: int = 0
    
    def __post_init__(self):
        self.current_node = self.origin


class TrafficSimulation:
    """
    Clase principal para ejecutar la simulación de tráfico.
    """
    
    def __init__(self, G: nx.MultiDiGraph, config_path: str = "config.yaml"):
        """
        Inicializa la simulación.
        
        Args:
            G: Grafo de la red vial
            config_path: Ruta al archivo de configuración
        """
        self.G = G.copy()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.sim_config = self.config['simulation']
        
        # Estado de la simulación
        self.current_time = 0.0
        self.time_step = self.sim_config['time_step']
        self.duration = self.sim_config['duration']
        
        # Agentes
        self.vehicles: List[Vehicle] = []
        self.active_vehicles: List[Vehicle] = []
        self.completed_vehicles: List[Vehicle] = []
        
        # Métricas
        self.edge_loads: Dict = {}  # Carga actual en cada arista
        self.history: List[Dict] = []  # Historia de la simulación
        
        # Inicializar cargas de aristas
        for u, v, k in self.G.edges(keys=True):
            self.edge_loads[(u, v, k)] = 0
        
        np.random.seed(self.config.get('random_seed', 42))
    
    def generate_od_pairs(self, num_agents: int) -> List[Tuple[int, int]]:
        """
        Genera pares origen-destino aleatorios.
        
        Args:
            num_agents: Número de pares OD a generar
            
        Returns:
            Lista de tuplas (origen, destino)
        """
        logger.info(f"Generando {num_agents} pares origen-destino...")
        
        nodes = list(self.G.nodes())
        od_pairs = []
        
        for _ in range(num_agents):
            origin, destination = np.random.choice(nodes, size=2, replace=False)
            od_pairs.append((origin, destination))
        
        return od_pairs
    
    def create_vehicles(self, od_pairs: List[Tuple[int, int]]):
        """
        Crea vehículos a partir de pares OD.
        
        Args:
            od_pairs: Lista de pares origen-destino
        """
        logger.info(f"Creando {len(od_pairs)} vehículos...")
        
        for i, (origin, destination) in enumerate(od_pairs):
            # Tiempo de salida distribuido durante la simulación
            departure_time = np.random.uniform(0, self.duration * 0.3)
            
            vehicle = Vehicle(
                id=i,
                origin=origin,
                destination=destination,
                current_node=origin,
                departure_time=departure_time
            )
            
            # Calcular ruta inicial
            self._calculate_route(vehicle)
            
            self.vehicles.append(vehicle)
    
    def _calculate_route(self, vehicle: Vehicle) -> bool:
        """
        Calcula la ruta para un vehículo.
        
        Args:
            vehicle: Vehículo para el cual calcular la ruta
            
        Returns:
            True si se encontró una ruta, False en caso contrario
        """
        try:
            # Usar el método de elección de ruta configurado
            if self.sim_config['route_choice'] == 'shortest_time':
                weight = 'travel_time'
            else:
                weight = 'length'
            
            path = nx.shortest_path(
                self.G, vehicle.origin, vehicle.destination, weight=weight
            )
            
            vehicle.path = path
            vehicle.path_index = 0
            vehicle.state = AgentState.WAITING
            
            return True
            
        except nx.NetworkXNoPath:
            logger.warning(f"No hay ruta entre {vehicle.origin} y {vehicle.destination}")
            vehicle.state = AgentState.ARRIVED
            return False
    
    def _should_reroute(self, vehicle: Vehicle) -> bool:
        """
        Determina si un vehículo debería re-enrutar.
        
        Args:
            vehicle: Vehículo a evaluar
            
        Returns:
            True si debe re-enrutar
        """
        if not self.sim_config['reroute_enabled']:
            return False
        
        if vehicle.path_index >= len(vehicle.path) - 1:
            return False
        
        # Obtener siguiente arista
        current = vehicle.path[vehicle.path_index]
        next_node = vehicle.path[vehicle.path_index + 1]
        
        # Buscar la arista en el multigrafo
        edge_data = None
        for key in self.G[current][next_node]:
            edge_data = self.G[current][next_node][key]
            break
        
        if edge_data is None:
            return False
        
        # Calcular congestión
        edge_key = (current, next_node, 0)
        current_load = self.edge_loads.get(edge_key, 0)
        
        # Si la carga es muy alta, considerar re-enrutamiento
        threshold_load = 10  # Vehículos por arista
        
        return current_load > threshold_load
    
    def _move_vehicle(self, vehicle: Vehicle):
        """
        Mueve un vehículo un paso en su ruta.
        
        Args:
            vehicle: Vehículo a mover
        """
        if vehicle.path_index >= len(vehicle.path) - 1:
            # El vehículo ha llegado
            vehicle.state = AgentState.ARRIVED
            vehicle.arrival_time = self.current_time
            vehicle.total_time = vehicle.arrival_time - vehicle.departure_time
            self.completed_vehicles.append(vehicle)
            return
        
        # Mover al siguiente nodo
        current = vehicle.path[vehicle.path_index]
        next_node = vehicle.path[vehicle.path_index + 1]
        
        # Obtener datos de la arista
        edge_data = self.G[current][next_node][0]
        
        # Actualizar posición
        vehicle.current_node = next_node
        vehicle.path_index += 1
        vehicle.total_distance += edge_data.get('length', 0)
        
        # Actualizar carga de aristas
        edge_key = (current, next_node, 0)
        self.edge_loads[edge_key] = self.edge_loads.get(edge_key, 0) + 1
        
        # Verificar si debe re-enrutar
        if self._should_reroute(vehicle):
            self._reroute_vehicle(vehicle)
    
    def _reroute_vehicle(self, vehicle: Vehicle):
        """
        Re-enruta un vehículo.
        
        Args:
            vehicle: Vehículo a re-enrutar
        """
        vehicle.reroute_count += 1
        vehicle.state = AgentState.REROUTING
        
        # Calcular nueva ruta desde posición actual
        original_dest = vehicle.destination
        vehicle.origin = vehicle.current_node
        
        self._calculate_route(vehicle)
        vehicle.destination = original_dest
    
    def step(self):
        """
        Ejecuta un paso de la simulación.
        """
        # Limpiar cargas de aristas
        for key in self.edge_loads:
            self.edge_loads[key] = 0
        
        # Activar vehículos que deben salir
        vehicles_to_activate = [
            v for v in self.vehicles 
            if v.departure_time <= self.current_time and v.state == AgentState.WAITING
        ]
        
        for vehicle in vehicles_to_activate:
            vehicle.state = AgentState.TRAVELING
            self.active_vehicles.append(vehicle)
        
        # Mover vehículos activos
        active_copy = self.active_vehicles.copy()
        for vehicle in active_copy:
            if vehicle.state == AgentState.TRAVELING or vehicle.state == AgentState.REROUTING:
                self._move_vehicle(vehicle)
                
                if vehicle.state == AgentState.ARRIVED:
                    self.active_vehicles.remove(vehicle)
        
        # Registrar estado
        self.history.append({
            'time': self.current_time,
            'active_vehicles': len(self.active_vehicles),
            'completed_vehicles': len(self.completed_vehicles),
            'total_vehicles': len(self.vehicles)
        })
        
        self.current_time += self.time_step
    
    def run(self, num_agents: Optional[int] = None, progress_bar: bool = True):
        """
        Ejecuta la simulación completa.
        
        Args:
            num_agents: Número de agentes a simular (usa config si es None)
            progress_bar: Si mostrar barra de progreso
        """
        if num_agents is None:
            num_agents = self.sim_config['num_agents']
        
        # Generar OD pairs y crear vehículos
        od_pairs = self.generate_od_pairs(num_agents)
        self.create_vehicles(od_pairs)
        
        logger.info(f"Iniciando simulación: {self.duration}s, {num_agents} vehículos")
        
        # Ejecutar simulación
        num_steps = int(self.duration / self.time_step)
        
        if progress_bar:
            pbar = tqdm(total=num_steps, desc="Simulando")
        
        while self.current_time < self.duration:
            self.step()
            
            if progress_bar:
                pbar.update(1)
        
        if progress_bar:
            pbar.close()
        
        logger.info(f"Simulación completada. {len(self.completed_vehicles)}/{num_agents} vehículos llegaron")
    
    def get_results(self) -> Dict:
        """
        Obtiene los resultados de la simulación.
        
        Returns:
            Diccionario con resultados y métricas
        """
        if not self.completed_vehicles:
            logger.warning("No hay vehículos completados")
            return {}
        
        # Calcular métricas
        travel_times = [v.total_time for v in self.completed_vehicles]
        distances = [v.total_distance for v in self.completed_vehicles]
        reroutes = [v.reroute_count for v in self.completed_vehicles]
        
        results = {
            'summary': {
                'total_vehicles': len(self.vehicles),
                'completed_vehicles': len(self.completed_vehicles),
                'completion_rate': len(self.completed_vehicles) / len(self.vehicles) * 100,
                'avg_travel_time_min': np.mean(travel_times) / 60,
                'std_travel_time_min': np.std(travel_times) / 60,
                'avg_distance_km': np.mean(distances) / 1000,
                'avg_reroutes': np.mean(reroutes),
                'total_reroutes': sum(reroutes)
            },
            'travel_times': travel_times,
            'distances': distances,
            'reroutes': reroutes,
            'history': pd.DataFrame(self.history),
            'vehicles': self.completed_vehicles
        }
        
        return results
    
    def save_results(self, filename: str = "simulation_results.pkl"):
        """
        Guarda los resultados de la simulación.
        
        Args:
            filename: Nombre del archivo
        """
        results = self.get_results()
        
        filepath = Path(self.config['paths']['results']) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Resultados guardados en {filepath}")


def main():
    """
    Función principal para ejecutar la simulación.
    """
    from src.data_acquisition import OSMDataAcquisition
    
    # Cargar la red
    acquisitor = OSMDataAcquisition()
    G = acquisitor.load_network()
    
    # Crear y ejecutar simulación
    sim = TrafficSimulation(G)
    sim.run(num_agents=500)
    
    # Obtener resultados
    results = sim.get_results()
    
    print("\n" + "="*50)
    print("RESULTADOS DE LA SIMULACIÓN")
    print("="*50)
    for key, value in results['summary'].items():
        print(f"{key}: {value:.2f}")
    print("="*50)
    
    # Guardar resultados
    sim.save_results()


if __name__ == "__main__":
    main()

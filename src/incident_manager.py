"""
Módulo para la gestión de incidentes en la red vial.

Este módulo maneja la creación y aplicación de diferentes tipos de incidentes
(accidentes, cierres de vías) durante la simulación.
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Optional
from enum import Enum
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IncidentType(Enum):
    """Tipos de incidentes posibles."""
    ACCIDENT = "accident"
    ROAD_CLOSURE = "road_closure"
    CONSTRUCTION = "construction"
    SPECIAL_EVENT = "special_event"


class Incident:
    """
    Clase que representa un incidente en la red.
    """
    
    def __init__(
        self,
        incident_type: IncidentType,
        affected_edges: List[Tuple],
        capacity_reduction: float,
        start_time: float,
        duration: float,
        description: str = ""
    ):
        """
        Inicializa un incidente.
        
        Args:
            incident_type: Tipo de incidente
            affected_edges: Lista de aristas afectadas (u, v, key)
            capacity_reduction: Fracción de reducción de capacidad (0.0 a 1.0)
            start_time: Tiempo de inicio del incidente
            duration: Duración del incidente en segundos
            description: Descripción del incidente
        """
        self.incident_type = incident_type
        self.affected_edges = affected_edges
        self.capacity_reduction = capacity_reduction
        self.start_time = start_time
        self.duration = duration
        self.end_time = start_time + duration
        self.description = description
        self.is_active = False
        self.original_edge_data = {}
    
    def is_currently_active(self, current_time: float) -> bool:
        """
        Verifica si el incidente está activo en el tiempo actual.
        
        Args:
            current_time: Tiempo actual de la simulación
            
        Returns:
            True si el incidente está activo
        """
        return self.start_time <= current_time < self.end_time
    
    def __repr__(self):
        return (f"Incident({self.incident_type.value}, "
                f"{len(self.affected_edges)} edges, "
                f"reduction={self.capacity_reduction:.1%})")


class IncidentManager:
    """
    Clase para gestionar incidentes en la simulación.
    """
    
    def __init__(self, G: nx.MultiDiGraph, config_path: str = "config.yaml"):
        """
        Inicializa el gestor de incidentes.
        
        Args:
            G: Grafo de la red vial
            config_path: Ruta al archivo de configuración
        """
        self.G = G
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.incidents: List[Incident] = []
        self.active_incidents: List[Incident] = []
        
        np.random.seed(self.config.get('random_seed', 42))
    
    def create_incident_from_scenario(
        self,
        scenario_name: str,
        start_time: float = 0.0
    ) -> Optional[Incident]:
        """
        Crea un incidente basado en un escenario predefinido.
        
        Args:
            scenario_name: Nombre del escenario en config
            start_time: Tiempo de inicio del incidente
            
        Returns:
            Objeto Incident o None si el escenario no existe
        """
        if scenario_name not in self.config['incidents']:
            logger.error(f"Escenario '{scenario_name}' no encontrado")
            return None
        
        scenario = self.config['incidents'][scenario_name]
        
        # Para baseline, no crear incidente
        if scenario_name == 'baseline':
            return None
        
        # Seleccionar aristas críticas aleatoriamente
        num_affected = scenario.get('num_affected', 1)
        affected_edges = self._select_critical_edges(num_affected)
        
        incident = Incident(
            incident_type=IncidentType.ACCIDENT,
            affected_edges=affected_edges,
            capacity_reduction=scenario['capacity_reduction'],
            start_time=start_time,
            duration=scenario['duration'],
            description=scenario['description']
        )
        
        self.incidents.append(incident)
        logger.info(f"Incidente creado: {incident}")
        
        return incident
    
    def _select_critical_edges(self, num_edges: int) -> List[Tuple]:
        """
        Selecciona aristas críticas de la red.
        
        Args:
            num_edges: Número de aristas a seleccionar
            
        Returns:
            Lista de aristas (u, v, key)
        """
        # Calcular betweenness de aristas para encontrar las críticas
        try:
            # Usar muestra para grafos grandes
            if len(self.G.nodes) > 500:
                k = min(100, len(self.G.nodes))
                edge_betweenness = nx.edge_betweenness_centrality(
                    self.G, k=k, weight='length', normalized=True
                )
            else:
                edge_betweenness = nx.edge_betweenness_centrality(
                    self.G, weight='length', normalized=True
                )
            
            # Ordenar aristas por betweenness
            sorted_edges = sorted(
                edge_betweenness.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Seleccionar de las top 20% más críticas
            top_critical = int(len(sorted_edges) * 0.2)
            critical_edges = [edge for edge, _ in sorted_edges[:top_critical]]
            
            # Seleccionar aleatoriamente del grupo crítico
            selected = np.random.choice(
                len(critical_edges),
                size=min(num_edges, len(critical_edges)),
                replace=False
            )
            
            result = [critical_edges[i] for i in selected]
            
            logger.info(f"Seleccionadas {len(result)} aristas críticas")
            return result
            
        except Exception as e:
            logger.error(f"Error seleccionando aristas críticas: {e}")
            # Fallback: seleccionar aristas aleatorias
            all_edges = list(self.G.edges(keys=True))
            return [all_edges[i] for i in np.random.choice(
                len(all_edges),
                size=min(num_edges, len(all_edges)),
                replace=False
            )]
    
    def apply_incident(self, incident: Incident):
        """
        Aplica un incidente al grafo, modificando las aristas afectadas.
        
        Args:
            incident: Incidente a aplicar
        """
        if incident.is_active:
            logger.warning(f"Incidente ya está activo: {incident}")
            return
        
        logger.info(f"Aplicando incidente: {incident}")
        
        for u, v, key in incident.affected_edges:
            try:
                # Guardar datos originales
                edge_id = (u, v, key)
                incident.original_edge_data[edge_id] = self.G[u][v][key].copy()
                
                # Modificar la arista según la reducción de capacidad
                edge_data = self.G[u][v][key]
                
                # Aumentar el tiempo de viaje proporcionalmente
                if 'travel_time' in edge_data:
                    original_time = edge_data['travel_time']
                    # A mayor reducción de capacidad, mayor tiempo de viaje
                    multiplier = 1 + (incident.capacity_reduction * 4)
                    edge_data['travel_time'] = original_time * multiplier
                
                # Si es un cierre total, podemos marcar la arista como no disponible
                if incident.capacity_reduction >= 1.0:
                    # Hacer el tiempo de viaje extremadamente alto
                    edge_data['travel_time'] = edge_data.get('travel_time', 100) * 1000
                
            except KeyError:
                logger.warning(f"Arista {u}-{v}-{key} no encontrada")
        
        incident.is_active = True
        self.active_incidents.append(incident)
    
    def remove_incident(self, incident: Incident):
        """
        Remueve un incidente del grafo, restaurando las aristas.
        
        Args:
            incident: Incidente a remover
        """
        if not incident.is_active:
            logger.warning(f"Incidente no está activo: {incident}")
            return
        
        logger.info(f"Removiendo incidente: {incident}")
        
        for edge_id, original_data in incident.original_edge_data.items():
            u, v, key = edge_id
            try:
                # Restaurar datos originales
                self.G[u][v][key].update(original_data)
            except KeyError:
                logger.warning(f"Arista {u}-{v}-{key} no encontrada al restaurar")
        
        incident.is_active = False
        self.active_incidents.remove(incident)
    
    def update_incidents(self, current_time: float):
        """
        Actualiza el estado de los incidentes basado en el tiempo actual.
        
        Args:
            current_time: Tiempo actual de la simulación
        """
        for incident in self.incidents:
            should_be_active = incident.is_currently_active(current_time)
            
            if should_be_active and not incident.is_active:
                self.apply_incident(incident)
            elif not should_be_active and incident.is_active:
                self.remove_incident(incident)
    
    def get_incident_info(self) -> Dict:
        """
        Obtiene información sobre todos los incidentes.
        
        Returns:
            Diccionario con información de incidentes
        """
        return {
            'total_incidents': len(self.incidents),
            'active_incidents': len(self.active_incidents),
            'incidents': [
                {
                    'type': inc.incident_type.value,
                    'description': inc.description,
                    'affected_edges': len(inc.affected_edges),
                    'capacity_reduction': inc.capacity_reduction,
                    'start_time': inc.start_time,
                    'duration': inc.duration,
                    'is_active': inc.is_active
                }
                for inc in self.incidents
            ]
        }


def main():
    """
    Función principal para probar el gestor de incidentes.
    """
    from src.data_acquisition import OSMDataAcquisition
    
    # Cargar la red
    acquisitor = OSMDataAcquisition()
    G = acquisitor.load_network()
    
    # Crear gestor de incidentes
    incident_mgr = IncidentManager(G)
    
    # Crear incidentes de prueba
    print("\n" + "="*50)
    print("CREANDO INCIDENTES DE PRUEBA")
    print("="*50)
    
    incident1 = incident_mgr.create_incident_from_scenario('incident_light', start_time=600)
    incident2 = incident_mgr.create_incident_from_scenario('road_closure', start_time=1200)
    
    # Mostrar información
    info = incident_mgr.get_incident_info()
    print(f"\nTotal de incidentes: {info['total_incidents']}")
    print(f"Incidentes activos: {info['active_incidents']}")
    
    for inc_info in info['incidents']:
        print(f"\n- {inc_info['description']}")
        print(f"  Tipo: {inc_info['type']}")
        print(f"  Aristas afectadas: {inc_info['affected_edges']}")
        print(f"  Reducción de capacidad: {inc_info['capacity_reduction']:.1%}")
        print(f"  Duración: {inc_info['duration']/60:.1f} minutos")


if __name__ == "__main__":
    main()

"""
Red Vial en Crisis - Proyecto de Modelación y Simulación
CC3039 - Universidad del Valle de Guatemala

Paquete principal para la simulación de redes viales urbanas.
"""

__version__ = "1.0.0"
__authors__ = ["Michelle Mejía Villela", "Silvia Illescas Fernandez"]

from . import data_acquisition
from . import network_model
from . import agent_simulation
from . import incident_manager
from . import metrics_analyzer
from . import visualization

__all__ = [
    'data_acquisition',
    'network_model',
    'agent_simulation',
    'incident_manager',
    'metrics_analyzer',
    'visualization'
]

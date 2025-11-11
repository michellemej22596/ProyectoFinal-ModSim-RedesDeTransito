"""
Script principal para ejecutar simulaciones completas.

Este script ejecuta simulaciones para diferentes escenarios y guarda los resultados.
"""

import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_acquisition import OSMDataAcquisition
from src.network_model import NetworkModel
from src.agent_simulation import TrafficSimulation
from src.incident_manager import IncidentManager
from src.metrics_analyzer import MetricsAnalyzer
from src.visualization import Visualizer
from src.ml_explainability import MLExplainabilityAnalyzer

import argparse
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_scenario(
    G,
    scenario_name: str,
    num_agents: int = 1000,
    duration: int = 3600
) -> dict:
    """
    Ejecuta una simulación para un escenario específico.
    
    Args:
        G: Grafo de la red vial
        scenario_name: Nombre del escenario
        num_agents: Número de agentes
        duration: Duración en segundos
        
    Returns:
        Diccionario con resultados
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"EJECUTANDO ESCENARIO: {scenario_name.upper()}")
    logger.info(f"{'='*70}\n")
    
    G_scenario = G.copy()
    
    # Crear instancia de gestor de incidentes CON EL GRAFO DEL ESCENARIO
    incident_mgr = IncidentManager(G_scenario)
    
    # Crear incidente si no es baseline
    if scenario_name != 'baseline':
        incident = incident_mgr.create_incident_from_scenario(
            scenario_name,
            start_time=0  # Aplicar incidente desde el inicio
        )
        # Aplicar el incidente inmediatamente
        if incident:
            incident_mgr.apply_incident(incident)
    
    sim = TrafficSimulation(G_scenario)
    
    sim.incident_manager = incident_mgr
    
    sim.run(num_agents=num_agents, progress_bar=True)
    
    # Obtener resultados
    results = sim.get_results()
    
    # Guardar resultados
    results_file = f"simulation_{scenario_name}.pkl"
    sim.save_results(results_file)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Ejecutar simulación de red vial'
    )
    parser.add_argument(
        '--scenario',
        type=str,
        default='all',
        choices=['all', 'baseline', 'incident_light', 'incident_moderate', 'road_closure'],
        help='Escenario a ejecutar'
    )
    parser.add_argument(
        '--agents',
        type=int,
        default=None,
        help='Número de agentes (vehículos)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=None,
        help='Duración de la simulación en segundos'
    )
    parser.add_argument(
        '--download',
        action='store_true',
        help='Descargar datos de OSM antes de simular'
    )
    parser.add_argument(
        '--ml-analysis',
        action='store_true',
        default=True,
        help='Ejecutar análisis ML con SHAP values (default: True)'
    )
    
    args = parser.parse_args()
    
    # Adquisición de datos
    acquisitor = OSMDataAcquisition()
    
    if args.download:
        logger.info("Descargando red desde OpenStreetMap...")
        G, _ = acquisitor.download_network()
    else:
        logger.info("Cargando red desde archivo...")
        G = acquisitor.load_network()
    
    import yaml
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    num_agents = args.agents if args.agents else config['simulation']['num_agents']
    duration = args.duration if args.duration else config['simulation']['duration']
    
    # Determinar escenarios a ejecutar
    if args.scenario == 'all':
        scenarios = ['baseline', 'incident_light', 'incident_moderate', 'road_closure']
    else:
        scenarios = [args.scenario]
    
    # Ejecutar escenarios
    all_results = {}
    
    for scenario in scenarios:
        results = run_scenario(
            G,
            scenario,
            num_agents=num_agents,
            duration=duration
        )
        all_results[scenario] = results
    
    # Análisis de métricas
    logger.info("\n" + "="*70)
    logger.info("ANALIZANDO RESULTADOS")
    logger.info("="*70 + "\n")
    
    analyzer = MetricsAnalyzer()
    
    # Generar reporte
    report = analyzer.generate_summary_report(all_results)
    print("\n" + report)
    
    # Guardar métricas
    analyzer.save_metrics(all_results)
    
    # Crear visualizaciones
    logger.info("\n" + "="*70)
    logger.info("GENERANDO VISUALIZACIONES")
    logger.info("="*70 + "\n")
    
    viz = Visualizer(G)
    
    # Mapa de la red
    from src.network_model import NetworkModel
    model = NetworkModel(G)
    critical_nodes_df = model.identify_critical_nodes(top_n=10)
    critical_edges_df = model.identify_critical_edges(top_n=10)
    
    viz.plot_network_map(
        critical_nodes=critical_nodes_df['node_id'].tolist(),
        critical_edges=list(zip(
            critical_edges_df['from_node'],
            critical_edges_df['to_node']
        ))
    )
    
    # Gráficas de resultados
    viz.plot_travel_time_distribution(all_results)
    viz.plot_vehicles_over_time(all_results)
    
    # Comparación de escenarios
    comparison_df = analyzer.compare_scenarios(all_results)
    viz.plot_scenario_comparison(comparison_df)
    
    if args.ml_analysis:
        logger.info("\n" + "="*70)
        logger.info("EJECUTANDO ANÁLISIS ML + SHAP")
        logger.info("="*70 + "\n")
        
        ml_analyzer = MLExplainabilityAnalyzer(output_dir="results/ml_analysis")
        ml_results = ml_analyzer.run_full_analysis(all_results)
        
        logger.info("\nAnálisis ML completado:")
        logger.info(f"  - R² Score: {ml_results['model_metrics']['test']['r2']:.4f}")
        logger.info(f"  - RMSE: {ml_results['model_metrics']['test']['rmse']:.2f} segundos")
        logger.info(f"  - Visualizaciones SHAP guardadas en: results/ml_analysis/")
    
    logger.info("\n" + "="*70)
    logger.info("SIMULACIÓN COMPLETADA")
    logger.info("="*70)
    logger.info(f"\nResultados guardados en: {Path('data/results').absolute()}")
    logger.info(f"Visualizaciones guardadas en: {Path('results/figures').absolute()}")


if __name__ == "__main__":
    main()

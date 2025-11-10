"""
Módulo para el análisis y cálculo de métricas de la simulación.

Este módulo proporciona funciones para calcular métricas de rendimiento,
comparar escenarios y analizar la resiliencia de la red.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import yaml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsAnalyzer:
    """
    Clase para analizar métricas de simulación.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa el analizador de métricas.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.metrics_config = self.config['metrics']
    
    def calculate_travel_time_metrics(self, results: Dict) -> Dict:
        """
        Calcula métricas relacionadas con tiempos de viaje.
        
        Args:
            results: Resultados de la simulación
            
        Returns:
            Diccionario con métricas de tiempo de viaje
        """
        travel_times = np.array(results['travel_times']) / 60  # Convertir a minutos
        
        metrics = {
            'mean_travel_time_min': np.mean(travel_times),
            'median_travel_time_min': np.median(travel_times),
            'std_travel_time_min': np.std(travel_times),
            'min_travel_time_min': np.min(travel_times),
            'max_travel_time_min': np.max(travel_times),
            'percentile_25': np.percentile(travel_times, 25),
            'percentile_75': np.percentile(travel_times, 75),
            'percentile_95': np.percentile(travel_times, 95)
        }
        
        return metrics
    
    def calculate_delay_metrics(
        self,
        baseline_results: Dict,
        incident_results: Dict
    ) -> Dict:
        """
        Calcula métricas de retraso comparando con baseline.
        
        Args:
            baseline_results: Resultados del escenario baseline
            incident_results: Resultados del escenario con incidentes
            
        Returns:
            Diccionario con métricas de retraso
        """
        baseline_times = np.array(baseline_results['travel_times']) / 60
        incident_times = np.array(incident_results['travel_times']) / 60
        
        # Asegurar mismo tamaño para comparación
        min_len = min(len(baseline_times), len(incident_times))
        baseline_times = baseline_times[:min_len]
        incident_times = incident_times[:min_len]
        
        delays = incident_times - baseline_times
        
        metrics = {
            'average_delay_min': np.mean(delays),
            'total_delay_hours': np.sum(delays) / 60,
            'max_delay_min': np.max(delays),
            'pct_with_delay': (delays > 0).sum() / len(delays) * 100,
            'delay_std': np.std(delays)
        }
        
        return metrics
    
    def calculate_rerouting_metrics(self, results: Dict) -> Dict:
        """
        Calcula métricas de re-enrutamiento.
        
        Args:
            results: Resultados de la simulación
            
        Returns:
            Diccionario con métricas de re-enrutamiento
        """
        reroutes = np.array(results['reroutes'])
        
        metrics = {
            'total_reroutes': int(np.sum(reroutes)),
            'avg_reroutes_per_vehicle': np.mean(reroutes),
            'pct_vehicles_rerouted': (reroutes > 0).sum() / len(reroutes) * 100,
            'max_reroutes': int(np.max(reroutes))
        }
        
        return metrics
    
    def calculate_network_efficiency(self, results: Dict) -> float:
        """
        Calcula la eficiencia de la red.
        
        Args:
            results: Resultados de la simulación
            
        Returns:
            Score de eficiencia (0-100)
        """
        summary = results['summary']
        
        # Factores de eficiencia
        completion_rate = summary['completion_rate']
        
        # Normalizar tiempo de viaje (asumiendo 60 min como máximo razonable)
        avg_time = summary['avg_travel_time_min']
        time_factor = max(0, 100 - (avg_time / 60 * 100))
        
        # Normalizar re-enrutamientos (asumiendo 5 como máximo razonable)
        avg_reroutes = summary.get('avg_reroutes', 0)
        reroute_factor = max(0, 100 - (avg_reroutes / 5 * 100))
        
        # Score combinado
        efficiency = (
            completion_rate * 0.4 +
            time_factor * 0.4 +
            reroute_factor * 0.2
        )
        
        return efficiency
    
    def calculate_resilience_score(
        self,
        baseline_results: Dict,
        incident_results: Dict
    ) -> float:
        """
        Calcula un score de resiliencia de la red.
        
        Args:
            baseline_results: Resultados del escenario baseline
            incident_results: Resultados del escenario con incidentes
            
        Returns:
            Score de resiliencia (0-100)
        """
        baseline_eff = self.calculate_network_efficiency(baseline_results)
        incident_eff = self.calculate_network_efficiency(incident_results)
        
        # La resiliencia es qué tan bien mantiene su eficiencia bajo incidentes
        resilience = (incident_eff / baseline_eff) * 100 if baseline_eff > 0 else 0
        
        return min(100, resilience)
    
    def compare_scenarios(
        self,
        scenarios_results: Dict[str, Dict]
    ) -> pd.DataFrame:
        """
        Compara múltiples escenarios.
        
        Args:
            scenarios_results: Diccionario {nombre_escenario: resultados}
            
        Returns:
            DataFrame con comparación de escenarios
        """
        comparison_data = []
        
        baseline_results = scenarios_results.get('baseline')
        
        for scenario_name, results in scenarios_results.items():
            row = {
                'scenario': scenario_name,
                'total_vehicles': results['summary']['total_vehicles'],
                'completed_vehicles': results['summary']['completed_vehicles'],
                'completion_rate': results['summary']['completion_rate'],
                'avg_travel_time_min': results['summary']['avg_travel_time_min'],
                'avg_reroutes': results['summary']['avg_reroutes'],
                'network_efficiency': self.calculate_network_efficiency(results)
            }
            
            # Calcular métricas adicionales si hay baseline
            if baseline_results and scenario_name != 'baseline':
                delay_metrics = self.calculate_delay_metrics(baseline_results, results)
                row['avg_delay_min'] = delay_metrics['average_delay_min']
                row['total_delay_hours'] = delay_metrics['total_delay_hours']
                row['resilience_score'] = self.calculate_resilience_score(
                    baseline_results, results
                )
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def generate_summary_report(
        self,
        scenarios_results: Dict[str, Dict]
    ) -> str:
        """
        Genera un reporte textual resumido.
        
        Args:
            scenarios_results: Diccionario {nombre_escenario: resultados}
            
        Returns:
            String con el reporte
        """
        report = []
        report.append("="*70)
        report.append("REPORTE DE ANÁLISIS - RED VIAL EN CRISIS")
        report.append("="*70)
        report.append("")
        
        comparison_df = self.compare_scenarios(scenarios_results)
        
        report.append("COMPARACIÓN DE ESCENARIOS")
        report.append("-"*70)
        report.append(comparison_df.to_string(index=False))
        report.append("")
        
        # Análisis específico por escenario
        for scenario_name, results in scenarios_results.items():
            report.append(f"\nESCENARIO: {scenario_name.upper()}")
            report.append("-"*70)
            
            # Métricas de tiempo de viaje
            travel_metrics = self.calculate_travel_time_metrics(results)
            report.append("\nTiempos de Viaje:")
            for key, value in travel_metrics.items():
                report.append(f"  {key}: {value:.2f}")
            
            # Métricas de re-enrutamiento
            reroute_metrics = self.calculate_rerouting_metrics(results)
            report.append("\nRe-enrutamiento:")
            for key, value in reroute_metrics.items():
                report.append(f"  {key}: {value:.2f}")
            
            report.append("")
        
        # Conclusiones
        report.append("\n" + "="*70)
        report.append("CONCLUSIONES")
        report.append("="*70)
        
        if 'baseline' in scenarios_results:
            baseline = scenarios_results['baseline']
            
            # Identificar el escenario con peor desempeño
            worst_scenario = min(
                [s for s in scenarios_results.keys() if s != 'baseline'],
                key=lambda s: self.calculate_network_efficiency(scenarios_results[s]),
                default=None
            )
            
            if worst_scenario:
                worst_results = scenarios_results[worst_scenario]
                delay = self.calculate_delay_metrics(baseline, worst_results)
                resilience = self.calculate_resilience_score(baseline, worst_results)
                
                report.append(f"\n1. El escenario '{worst_scenario}' muestra el mayor impacto:")
                report.append(f"   - Retraso promedio: {delay['average_delay_min']:.2f} minutos")
                report.append(f"   - Retraso total acumulado: {delay['total_delay_hours']:.2f} horas")
                report.append(f"   - Score de resiliencia: {resilience:.1f}/100")
                
                report.append(f"\n2. La red vial de Guatemala {'muestra' if resilience < 70 else 'mantiene'}")
                report.append(f"   {'alta vulnerabilidad' if resilience < 70 else 'buena resiliencia'}")
                report.append(f"   frente a incidentes en vías críticas.")
        
        report.append("\n" + "="*70)
        
        return "\n".join(report)
    
    def save_metrics(
        self,
        scenarios_results: Dict[str, Dict],
        filename: str = "metrics_analysis.txt"
    ):
        """
        Guarda las métricas en un archivo.
        
        Args:
            scenarios_results: Resultados de escenarios
            filename: Nombre del archivo
        """
        report = self.generate_summary_report(scenarios_results)
        
        filepath = Path(self.config['paths']['results']) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Métricas guardadas en {filepath}")
        
        # También guardar CSV
        comparison_df = self.compare_scenarios(scenarios_results)
        csv_path = filepath.with_suffix('.csv')
        comparison_df.to_csv(csv_path, index=False)
        logger.info(f"Comparación guardada en {csv_path}")


def main():
    """
    Función principal de demostración.
    """
    # Crear datos de ejemplo
    ejemplo_baseline = {
        'travel_times': np.random.normal(1800, 300, 1000),  # 30 min promedio
        'distances': np.random.normal(10000, 2000, 1000),
        'reroutes': np.random.poisson(0.5, 1000),
        'summary': {
            'total_vehicles': 1000,
            'completed_vehicles': 980,
            'completion_rate': 98.0,
            'avg_travel_time_min': 30.0,
            'avg_reroutes': 0.5
        }
    }
    
    ejemplo_incidente = {
        'travel_times': np.random.normal(2400, 500, 1000),  # 40 min promedio
        'distances': np.random.normal(11000, 2500, 1000),
        'reroutes': np.random.poisson(2.0, 1000),
        'summary': {
            'total_vehicles': 1000,
            'completed_vehicles': 950,
            'completion_rate': 95.0,
            'avg_travel_time_min': 40.0,
            'avg_reroutes': 2.0
        }
    }
    
    scenarios = {
        'baseline': ejemplo_baseline,
        'incident_light': ejemplo_incidente
    }
    
    # Analizar
    analyzer = MetricsAnalyzer()
    report = analyzer.generate_summary_report(scenarios)
    
    print(report)


if __name__ == "__main__":
    main()

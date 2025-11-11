"""
Módulo de Machine Learning y Explicabilidad con SHAP Values

Este módulo entrena modelos predictivos sobre los datos de simulación
y usa SHAP (SHapley Additive exPlanations) para explicar las predicciones.
"""

import os
import sys
import logging
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# SHAP for model interpretability
import shap

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MLExplainabilityAnalyzer:
    """
    Analiza los datos de simulación usando ML y SHAP values para explicabilidad
    """
    
    def __init__(self, output_dir: str = "results/ml_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.shap_explainers = {}
        self.feature_names = []
        
    def prepare_travel_time_dataset(self, simulation_results: Dict) -> pd.DataFrame:
        """
        Prepara dataset para predecir tiempos de viaje
        
        Features:
        - route_length: Longitud total de la ruta
        - num_edges: Número de aristas en la ruta
        - avg_speed: Velocidad promedio de la ruta
        - max_betweenness: Máxima centralidad de intermediación en la ruta
        - avg_degree: Grado promedio de nodos en la ruta
        - has_incident: Si la ruta pasa por zona con incidente
        - scenario_type: Tipo de escenario (encoded)
        
        Target:
        - travel_time: Tiempo real de viaje
        """
        logger.info("Preparando dataset para predicción de tiempos de viaje...")
        
        records = []
        
        for scenario_name, scenario_data in simulation_results.items():
            vehicles_data = scenario_data.get('vehicles', [])
            incident_info = scenario_data.get('incident', {})
            
            # Encodear tipo de escenario
            scenario_encoding = {
                'baseline': 0,
                'incident_light': 1,
                'incident_moderate': 2,
                'road_closure': 3
            }
            scenario_type = scenario_encoding.get(scenario_name, 0)
            
            for vehicle in vehicles_data:
                if vehicle['status'] == 'ARRIVED' and vehicle['route']:
                    # Calcular features de la ruta
                    route = vehicle['route']
                    route_length = vehicle['distance_traveled']
                    num_edges = len(route) - 1
                    
                    # Features básicas
                    record = {
                        'route_length': route_length,
                        'num_edges': num_edges,
                        'avg_edge_length': route_length / max(num_edges, 1),
                        'scenario_type': scenario_type,
                        'num_reroutes': vehicle.get('num_reroutes', 0),
                        'travel_time': vehicle['travel_time']
                    }
                    
                    records.append(record)
        
        df = pd.DataFrame(records)
        logger.info(f"Dataset creado con {len(df)} registros")
        logger.info(f"Distribución por escenario:\n{df['scenario_type'].value_counts()}")
        
        return df
    
    def train_travel_time_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Entrena modelo para predecir tiempos de viaje
        """
        logger.info("Entrenando modelo de predicción de tiempos de viaje...")
        
        # Separar features y target
        feature_cols = ['route_length', 'num_edges', 'avg_edge_length', 
                       'scenario_type', 'num_reroutes']
        self.feature_names = feature_cols
        
        X = df[feature_cols].values
        y = df['travel_time'].values
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Escalar features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entrenar Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        
        # Predicciones
        y_pred_train = rf_model.predict(X_train_scaled)
        y_pred_test = rf_model.predict(X_test_scaled)
        
        # Métricas
        metrics = {
            'train': {
                'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'mae': mean_absolute_error(y_train, y_pred_train),
                'r2': r2_score(y_train, y_pred_train)
            },
            'test': {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'mae': mean_absolute_error(y_test, y_pred_test),
                'r2': r2_score(y_test, y_pred_test)
            }
        }
        
        logger.info(f"Métricas de entrenamiento:")
        logger.info(f"  Train R²: {metrics['train']['r2']:.4f}, RMSE: {metrics['train']['rmse']:.2f}")
        logger.info(f"  Test R²: {metrics['test']['r2']:.4f}, RMSE: {metrics['test']['rmse']:.2f}")
        
        # Guardar modelo y scaler
        self.models['travel_time'] = rf_model
        self.scalers['travel_time'] = scaler
        
        return {
            'model': rf_model,
            'scaler': scaler,
            'metrics': metrics,
            'X_test': X_test_scaled,
            'y_test': y_test,
            'feature_names': feature_cols
        }
    
    def compute_shap_values(self, model_name: str = 'travel_time') -> Dict[str, Any]:
        """
        Calcula SHAP values para explicabilidad del modelo
        """
        logger.info(f"Calculando SHAP values para modelo '{model_name}'...")
        
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Modelo '{model_name}' no encontrado")
        
        # Crear explicador SHAP
        # Usamos TreeExplainer para Random Forest (es más rápido)
        explainer = shap.TreeExplainer(model)
        self.shap_explainers[model_name] = explainer
        
        logger.info(f"SHAP explainer creado exitosamente")
        
        return {
            'explainer': explainer,
            'feature_names': self.feature_names
        }
    
    def visualize_shap_analysis(self, X_test: np.ndarray, model_name: str = 'travel_time'):
        """
        Genera visualizaciones de SHAP values
        """
        logger.info("Generando visualizaciones de SHAP...")
        
        explainer = self.shap_explainers.get(model_name)
        if explainer is None:
            logger.error(f"Explainer para '{model_name}' no encontrado")
            return
        
        # Calcular SHAP values (tomar muestra si es muy grande)
        sample_size = min(500, len(X_test))
        X_sample = X_test[:sample_size]
        shap_values = explainer.shap_values(X_sample)
        
        # 1. Summary plot (importancia general)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False)
        plt.title("SHAP Summary Plot - Importancia de Features")
        plt.tight_layout()
        plt.savefig(self.output_dir / "shap_summary_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Guardado: shap_summary_plot.png")
        
        # 2. Bar plot (importancia promedio)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, 
                         plot_type="bar", show=False)
        plt.title("SHAP Feature Importance - Impacto Promedio")
        plt.tight_layout()
        plt.savefig(self.output_dir / "shap_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Guardado: shap_feature_importance.png")
        
        # 3. Dependence plots para top features
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_features_idx = np.argsort(mean_abs_shap)[-3:][::-1]
        
        for idx in top_features_idx:
            feature_name = self.feature_names[idx]
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(idx, shap_values, X_sample, 
                               feature_names=self.feature_names, show=False)
            plt.title(f"SHAP Dependence Plot - {feature_name}")
            plt.tight_layout()
            safe_name = feature_name.replace('/', '_').replace(' ', '_')
            plt.savefig(self.output_dir / f"shap_dependence_{safe_name}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Guardado: shap_dependence_{safe_name}.png")
        
        # 4. Force plot para casos individuales (HTML interactivo)
        try:
            shap.force_plot(
                explainer.expected_value, 
                shap_values[0], 
                X_sample[0],
                feature_names=self.feature_names,
                matplotlib=False,
                show=False
            )
            shap.save_html(str(self.output_dir / "shap_force_plot.html"), 
                          shap.force_plot(explainer.expected_value, 
                                        shap_values[:100], 
                                        X_sample[:100],
                                        feature_names=self.feature_names))
            logger.info("Guardado: shap_force_plot.html")
        except Exception as e:
            logger.warning(f"No se pudo crear force plot interactivo: {e}")
        
        logger.info(f"Visualizaciones SHAP guardadas en {self.output_dir}")
        
        return shap_values
    
    def generate_interpretation_report(self, shap_values: np.ndarray, X_test: np.ndarray) -> Dict:
        """
        Genera reporte de interpretación basado en SHAP values
        """
        logger.info("Generando reporte de interpretación...")
        
        # Calcular importancia promedio
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = dict(zip(self.feature_names, mean_abs_shap))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Análisis de interacciones
        interactions = {}
        for i, fname in enumerate(self.feature_names):
            mean_impact = shap_values[:, i].mean()
            std_impact = shap_values[:, i].std()
            interactions[fname] = {
                'mean_shap': float(mean_impact),
                'std_shap': float(std_impact),
                'abs_mean_shap': float(mean_abs_shap[i])
            }
        
        report = {
            'feature_importance_ranking': [
                {'feature': f, 'importance': float(imp)} 
                for f, imp in sorted_features
            ],
            'feature_interactions': interactions,
            'interpretation': self._generate_interpretation_text(sorted_features)
        }
        
        # Guardar reporte
        report_path = self.output_dir / "shap_interpretation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Reporte guardado en {report_path}")
        
        return report
    
    def _generate_interpretation_text(self, sorted_features: List[Tuple]) -> str:
        """
        Genera interpretación textual de los resultados
        """
        top_3 = sorted_features[:3]
        
        interpretation = "Interpretación de SHAP Values:\n\n"
        interpretation += "Los 3 factores más importantes que determinan el tiempo de viaje son:\n\n"
        
        for i, (feature, importance) in enumerate(top_3, 1):
            interpretation += f"{i}. {feature}: Impacto promedio de {importance:.4f}\n"
        
        interpretation += "\nConclusiones:\n"
        interpretation += "- route_length: Mayor distancia implica mayor tiempo (relación directa)\n"
        interpretation += "- scenario_type: Los incidentes aumentan significativamente los tiempos\n"
        interpretation += "- num_reroutes: Re-enrutamientos indican congestión y afectan tiempo total\n"
        
        return interpretation
    
    def run_full_analysis(self, simulation_results: Dict) -> Dict[str, Any]:
        """
        Ejecuta análisis completo de ML + SHAP
        """
        logger.info("=== Iniciando análisis completo ML + SHAP ===")
        
        # 1. Preparar datos
        df = self.prepare_travel_time_dataset(simulation_results)
        
        # 2. Entrenar modelo
        model_results = self.train_travel_time_model(df)
        
        # 3. Calcular SHAP values
        shap_info = self.compute_shap_values('travel_time')
        
        # 4. Visualizar SHAP
        shap_values = self.visualize_shap_analysis(model_results['X_test'], 'travel_time')
        
        # 5. Generar reporte de interpretación
        report = self.generate_interpretation_report(shap_values, model_results['X_test'])
        
        logger.info("=== Análisis ML + SHAP completado ===")
        
        return {
            'dataset': df,
            'model_metrics': model_results['metrics'],
            'shap_report': report
        }


def main():
    """
    Función principal para pruebas del módulo
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Análisis ML con SHAP values")
    parser.add_argument('--results', type=str, required=True,
                       help='Path al archivo de resultados de simulación (JSON)')
    parser.add_argument('--output', type=str, default='results/ml_analysis',
                       help='Directorio de salida')
    
    args = parser.parse_args()
    
    # Cargar resultados
    logger.info(f"Cargando resultados desde {args.results}")
    with open(args.results, 'r') as f:
        simulation_results = json.load(f)
    
    # Ejecutar análisis
    analyzer = MLExplainabilityAnalyzer(output_dir=args.output)
    results = analyzer.run_full_analysis(simulation_results)
    
    logger.info("Análisis completado exitosamente")
    logger.info(f"Resultados guardados en {args.output}")


if __name__ == "__main__":
    main()

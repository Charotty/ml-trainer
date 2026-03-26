#!/usr/bin/env python3
"""
Target-Specific Ensemble Models for Kidney Displacement Prediction
Phase 2: Different ensemble combinations for different displacement axes
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class TargetSpecificEnsembleTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = []
        self.target_names = []
        self.results = {}
        
        self.required_features = [
            'kidney_left_center_x_rel', 'kidney_left_center_y_rel', 'kidney_left_center_z_rel',
            'kidney_left_center_x_norm', 'kidney_left_center_y_norm', 'kidney_left_center_z_norm',
            'kidney_right_center_x_rel', 'kidney_right_center_y_rel', 'kidney_right_center_z_rel',
            'kidney_right_center_x_norm', 'kidney_right_center_y_norm', 'kidney_right_center_z_norm',
            'kidney_left_length_mm', 'kidney_left_volume_cm3',
            'kidney_right_length_mm', 'kidney_right_volume_cm3',
            'body_width_mm', 'body_depth_mm', 'body_area_mm2',
            'kidney_left_to_spine_distance', 'kidney_right_to_spine_distance',
            'kidney_left_to_body_center_distance', 'kidney_right_to_body_center_distance',
            'spine_center_x', 'spine_center_y', 'spine_center_z',
            'body_com_x', 'body_com_y', 'body_com_z',
            'patient_position_encoded',
        ]

        self.target_columns = [
            'kidney_left_delta_x', 'kidney_left_delta_y', 'kidney_left_delta_z',
            'kidney_right_delta_x', 'kidney_right_delta_y', 'kidney_right_delta_z'
        ]
        
        # Target-specific ensemble configurations
        # Based on analysis: large displacement axes need different models than small displacement axes
        self.target_specific_configs = {
            # Large displacement axes (>4mm) - need robust models
            'kidney_left_delta_x': {
                'models': ['RandomForest', 'Ridge'],  # Best for large displacements
                'weights': [2.0, 1.0],
                'description': 'Large displacement axis - RandomForest + Ridge'
            },
            'kidney_right_delta_x': {
                'models': ['Ridge', 'RandomForest'],  # Ridge was best single
                'weights': [2.0, 1.0],
                'description': 'Large displacement axis - Ridge + RandomForest'
            },
            
            # Small displacement axes (<2.5mm) - need precise models
            'kidney_left_delta_y': {
                'models': ['RandomForest', 'Lasso'],  # Best for small displacements
                'weights': [1.5, 1.0],
                'description': 'Small displacement axis - RandomForest + Lasso'
            },
            'kidney_left_delta_z': {
                'models': ['Lasso', 'GradientBoosting'],  # Lasso was best single
                'weights': [2.0, 1.0],
                'description': 'Small displacement axis - Lasso + GradientBoosting'
            },
            'kidney_right_delta_y': {
                'models': ['RandomForest', 'Lasso'],  # Best for small displacements
                'weights': [2.0, 1.0],
                'description': 'Small displacement axis - RandomForest + Lasso'
            },
            'kidney_right_delta_z': {
                'models': ['GradientBoosting', 'RandomForest'],  # GradientBoosting was best single
                'weights': [2.0, 1.0],
                'description': 'Small displacement axis - GradientBoosting + RandomForest'
            }
        }
        
        # Best single models for comparison
        self.best_single_models = {
            'kidney_left_delta_x': 'RandomForest',
            'kidney_left_delta_y': 'RandomForest', 
            'kidney_left_delta_z': 'Lasso',
            'kidney_right_delta_x': 'Ridge',
            'kidney_right_delta_y': 'RandomForest',
            'kidney_right_delta_z': 'GradientBoosting'
        }
        
        # Best single model MAEs for comparison
        self.best_single_maes = {
            'kidney_left_delta_x': 4.022,
            'kidney_left_delta_y': 2.293,
            'kidney_left_delta_z': 2.366,
            'kidney_right_delta_x': 4.077,
            'kidney_right_delta_y': 1.907,
            'kidney_right_delta_z': 1.777
        }
    
    def load_and_prepare_data(self):
        """Load and prepare datasets for training"""
        print("Loading datasets...")
        
        # Load datasets
        vybor_df = pd.read_csv('data/vybor_unified_features.csv')
        kits19_df = pd.read_csv('data/kits19_medical_grade_features.csv')
        
        print(f"Vybor dataset: {len(vybor_df)} cases")
        print(f"KiTS19 dataset: {len(kits19_df)} cases")
        
        # Process datasets
        combined_data = []
        
        # Process vybor dataset
        if 'kidney_left_delta_x' in vybor_df.columns:
            vybor_processed = self.process_dataset(vybor_df, 'vybor')
            if vybor_processed is not None:
                combined_data.append(vybor_processed)
        
        # Process kits19 dataset
        if 'kidney_left_delta_x' in kits19_df.columns:
            kits19_processed = self.process_dataset(kits19_df, 'kits19')
            if kits19_processed is not None:
                combined_data.append(kits19_processed)
        
        if combined_data:
            final_df = pd.concat(combined_data, ignore_index=True)

            # Unify schema across datasets
            for c in self.required_features:
                if c not in final_df.columns:
                    final_df[c] = np.nan

            feature_cols = list(self.required_features)
            target_cols = [c for c in self.target_columns if c in final_df.columns]

            # Ensure targets are present; features are imputed after concat
            final_df = final_df.dropna(subset=target_cols)
            final_df[feature_cols] = self.imputer.fit_transform(final_df[feature_cols])

            print(f"Combined dataset: {len(final_df)} cases")
            return final_df
        else:
            return None
    
    def process_dataset(self, df, dataset_name):
        """Process dataset to extract features and targets"""
        available_features = [col for col in self.required_features if col in df.columns and col != 'patient_position_encoded']
        available_targets = [col for col in self.target_columns if col in df.columns]
        
        if len(available_targets) == 0:
            print(f"No target variables found in {dataset_name} dataset")
            return None
        
        cols = ['case_id'] + available_features + available_targets
        processed_df = df[cols].copy()

        # Patient position encoding: 0=supine, 1=other
        position_col = 'patient_position' if 'patient_position' in df.columns else 'scan_position'
        if position_col in df.columns:
            pos = df[position_col].fillna('supine').astype(str).str.lower()
            processed_df['patient_position_encoded'] = (~pos.str.contains('sup')).astype(int)
        else:
            processed_df['patient_position_encoded'] = 0

        # Add missing feature columns so concat schema is consistent; impute later
        for c in self.required_features:
            if c not in processed_df.columns:
                processed_df[c] = np.nan

        processed_df = processed_df.dropna(subset=available_targets)

        print(f"{dataset_name}: {len(processed_df)} cases")
        return processed_df
    
    def prepare_training_data(self, df):
        """Prepare features and targets for training"""
        target_cols = [col for col in self.target_columns if col in df.columns]
        feature_cols = list(self.required_features)
        
        X = df[feature_cols].values
        y = df[target_cols].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.feature_names = feature_cols
        self.target_names = target_cols
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def load_base_models(self):
        """Load base models with optimal parameters"""
        print("\nLoading base models with optimal parameters...")
        
        models = {}
        
        # Use best parameters found during optimization
        model_configs = {
            'RandomForest': {
                'n_estimators': 500,
                'max_depth': 20,
                'min_samples_split': 10,
                'min_samples_leaf': 4,
                'max_features': 'sqrt',
                'random_state': 42,
                'n_jobs': -1
            },
            'Lasso': {
                'alpha': 0.1,
                'max_iter': 5000,
                'random_state': 42
            },
            'Ridge': {
                'alpha': 1.0,
                'solver': 'auto',
                'random_state': 42
            },
            'GradientBoosting': {
                'n_estimators': 500,
                'learning_rate': 0.05,
                'max_depth': 5,
                'subsample': 0.8,
                'random_state': 42
            }
        }
        
        for model_name, config in model_configs.items():
            if model_name == 'RandomForest':
                models[model_name] = RandomForestRegressor(**config)
            elif model_name == 'Lasso':
                models[model_name] = Lasso(**config)
            elif model_name == 'Ridge':
                models[model_name] = Ridge(**config)
            elif model_name == 'GradientBoosting':
                models[model_name] = GradientBoostingRegressor(**config)
        
        return models
    
    def create_target_specific_ensemble(self, models, target_name):
        """Create target-specific ensemble for specific target"""
        config = self.target_specific_configs[target_name]
        model_names = config['models']
        weights = config['weights']
        
        # Create estimators
        estimators = []
        for model_name in model_names:
            if model_name in models:
                estimators.append((model_name, models[model_name]))
        
        return VotingRegressor(
            estimators=estimators,
            weights=weights,
            n_jobs=-1
        )
    
    def create_best_single_model(self, models, target_name):
        """Create the best single model for comparison"""
        best_model_name = self.best_single_models[target_name]
        return models[best_model_name]
    
    def evaluate_model_cv(self, model, X_train, y_train, model_name):
        """Evaluate model using cross-validation"""
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                  scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"  {model_name} CV MAE: {cv_mae:.3f} ± {cv_std:.3f}")
        return cv_mae, cv_std
    
    def train_and_evaluate_target_specific_ensembles(self, X_train, X_test, y_train, y_test):
        """Train and evaluate target-specific ensemble models"""
        print("\nTraining and evaluating target-specific ensemble models...")
        
        # Load base models
        base_models = self.load_base_models()
        
        results = {}
        
        for i, target_name in enumerate(self.target_names):
            print(f"\n{target_name}:")
            print("-" * 50)
            print(f"  Strategy: {self.target_specific_configs[target_name]['description']}")
            
            y_train_target = y_train[:, i]
            y_test_target = y_test[:, i]
            
            # Create target-specific ensemble
            target_specific_ensemble = self.create_target_specific_ensemble(base_models, target_name)
            
            # Create best single model for comparison
            best_single_model = self.create_best_single_model(base_models, target_name)
            
            # Train models
            target_specific_ensemble.fit(X_train, y_train_target)
            best_single_model.fit(X_train, y_train_target)
            
            # Predictions
            ensemble_pred = target_specific_ensemble.predict(X_test)
            single_pred = best_single_model.predict(X_test)
            
            # Metrics for target-specific ensemble
            ensemble_mae = mean_absolute_error(y_test_target, ensemble_pred)
            ensemble_rmse = np.sqrt(mean_squared_error(y_test_target, ensemble_pred))
            ensemble_r2 = r2_score(y_test_target, ensemble_pred)
            ensemble_median_ae = np.median(np.abs(y_test_target - ensemble_pred))
            
            # Clinical metrics for ensemble
            ensemble_error_5mm = np.mean(np.abs(y_test_target - ensemble_pred) < 5) * 100
            ensemble_error_10mm = np.mean(np.abs(y_test_target - ensemble_pred) < 10) * 100
            ensemble_max_error = np.max(np.abs(y_test_target - ensemble_pred))
            ensemble_outliers = np.sum(np.abs(y_test_target - ensemble_pred) > 20)
            ensemble_std_error = np.std(y_test_target - ensemble_pred)
            
            # Metrics for best single model
            single_mae = mean_absolute_error(y_test_target, single_pred)
            single_rmse = np.sqrt(mean_squared_error(y_test_target, single_pred))
            single_r2 = r2_score(y_test_target, single_pred)
            single_median_ae = np.median(np.abs(y_test_target - single_pred))
            
            # Clinical metrics for single
            single_error_5mm = np.mean(np.abs(y_test_target - single_pred) < 5) * 100
            single_error_10mm = np.mean(np.abs(y_test_target - single_pred) < 10) * 100
            single_max_error = np.max(np.abs(y_test_target - single_pred))
            single_outliers = np.sum(np.abs(y_test_target - single_pred) > 20)
            single_std_error = np.std(y_test_target - single_pred)
            
            # CV evaluation of models in ensemble
            print("  Ensemble Models CV Performance:")
            for model_name in self.target_specific_configs[target_name]['models']:
                if model_name in base_models:
                    cv_mae, cv_std = self.evaluate_model_cv(
                        base_models[model_name], X_train, y_train_target, model_name
                    )
            
            results[target_name] = {
                'TargetSpecific_MAE': ensemble_mae,
                'TargetSpecific_RMSE': ensemble_rmse,
                'TargetSpecific_R2': ensemble_r2,
                'TargetSpecific_Median_AE': ensemble_median_ae,
                'TargetSpecific_Error_5mm': ensemble_error_5mm,
                'TargetSpecific_Error_10mm': ensemble_error_10mm,
                'TargetSpecific_Max_Error': ensemble_max_error,
                'TargetSpecific_Outliers_20mm': ensemble_outliers,
                'TargetSpecific_Std_Error': ensemble_std_error,
                'BestSingle_MAE': single_mae,
                'BestSingle_RMSE': single_rmse,
                'BestSingle_R2': single_r2,
                'BestSingle_Median_AE': single_median_ae,
                'BestSingle_Error_5mm': single_error_5mm,
                'BestSingle_Error_10mm': single_error_10mm,
                'BestSingle_Max_Error': single_max_error,
                'BestSingle_Outliers_20mm': single_outliers,
                'BestSingle_Std_Error': single_std_error,
                'Best_Single_Model': self.best_single_models[target_name],
                'Improvement': ((self.best_single_maes[target_name] - ensemble_mae) / self.best_single_maes[target_name]) * 100,
                'Strategy': self.target_specific_configs[target_name]['description']
            }
            
            print(f"  Target-Specific Ensemble - MAE: {ensemble_mae:.3f} mm, R²: {ensemble_r2:.3f}")
            print(f"    <5mm accuracy: {ensemble_error_5mm:.1f}%, <10mm accuracy: {ensemble_error_10mm:.1f}%")
            print(f"  Best Single Model - MAE: {single_mae:.3f} mm, R²: {single_r2:.3f}")
            print(f"    <5mm accuracy: {single_error_5mm:.1f}%, <10mm accuracy: {single_error_10mm:.1f}%")
            print(f"  Improvement over best single: {((self.best_single_maes[target_name] - ensemble_mae) / self.best_single_maes[target_name]) * 100:.1f}%")
        
        self.results = results
        return results
    
    def generate_report(self):
        """Generate comprehensive target-specific ensemble report"""
        print("\n" + "="*80)
        print("TARGET-SPECIFIC ENSEMBLE MODELS - PHASE 2 REPORT")
        print("="*80)
        
        print(f"\nDataset Summary:")
        print(f"- Features used: {len(self.feature_names)}")
        print(f"- Target variables: {len(self.target_names)}")
        
        print(f"\nOverall Performance Summary:")
        print("-" * 50)
        
        # Calculate average metrics for target-specific ensembles
        ensemble_mae = np.mean([r['TargetSpecific_MAE'] for r in self.results.values()])
        ensemble_rmse = np.mean([r['TargetSpecific_RMSE'] for r in self.results.values()])
        ensemble_r2 = np.mean([r['TargetSpecific_R2'] for r in self.results.values()])
        ensemble_5mm = np.mean([r['TargetSpecific_Error_5mm'] for r in self.results.values()])
        ensemble_10mm = np.mean([r['TargetSpecific_Error_10mm'] for r in self.results.values()])
        
        # Calculate average metrics for best single models
        single_mae = np.mean([r['BestSingle_MAE'] for r in self.results.values()])
        single_rmse = np.mean([r['BestSingle_RMSE'] for r in self.results.values()])
        single_r2 = np.mean([r['BestSingle_R2'] for r in self.results.values()])
        single_5mm = np.mean([r['BestSingle_Error_5mm'] for r in self.results.values()])
        single_10mm = np.mean([r['BestSingle_Error_10mm'] for r in self.results.values()])
        
        print(f"Target-Specific Ensembles:")
        print(f"  Average MAE: {ensemble_mae:.3f} mm")
        print(f"  Average RMSE: {ensemble_rmse:.3f} mm")
        print(f"  Average R²: {ensemble_r2:.3f}")
        print(f"  Average <5mm accuracy: {ensemble_5mm:.1f}%")
        print(f"  Average <10mm accuracy: {ensemble_10mm:.1f}%")
        
        print(f"\nBest Single Models:")
        print(f"  Average MAE: {single_mae:.3f} mm")
        print(f"  Average RMSE: {single_rmse:.3f} mm")
        print(f"  Average R²: {single_r2:.3f}")
        print(f"  Average <5mm accuracy: {single_5mm:.1f}%")
        print(f"  Average <10mm accuracy: {single_10mm:.1f}%")
        
        print(f"\nImprovement Analysis:")
        avg_improvement = np.mean([r['Improvement'] for r in self.results.values()])
        improved_count = sum(1 for r in self.results.values() if r['Improvement'] > 0)
        
        print(f"  Average improvement: {avg_improvement:.1f}%")
        print(f"  Improved targets: {improved_count}/{len(self.results)}")
        print(f"  Success rate: {(improved_count/len(self.results)*100):.1f}%")
        
        print(f"\nDetailed Results by Target:")
        print("-" * 50)
        
        for target_name, metrics in self.results.items():
            print(f"\n{target_name}:")
            print(f"  Strategy: {metrics['Strategy']}")
            print(f"  Target-Specific Ensemble - MAE: {metrics['TargetSpecific_MAE']:.3f} mm (R²: {metrics['TargetSpecific_R2']:.3f})")
            print(f"  Best Single Model - MAE: {metrics['BestSingle_MAE']:.3f} mm (R²: {metrics['BestSingle_R2']:.3f})")
            print(f"  Improvement: {metrics['Improvement']:.1f}%")
    
    def save_results(self, filename='target_specific_ensemble_results.csv'):
        """Save results to CSV"""
        rows = []
        for target_name, metrics in self.results.items():
            # Target-specific ensemble row
            ensemble_row = {
                'Target': target_name,
                'Model': 'TargetSpecific_Ensemble',
                'MAE': metrics['TargetSpecific_MAE'],
                'RMSE': metrics['TargetSpecific_RMSE'],
                'R2': metrics['TargetSpecific_R2'],
                'Median_AE': metrics['TargetSpecific_Median_AE'],
                'Error_5mm': metrics['TargetSpecific_Error_5mm'],
                'Error_10mm': metrics['TargetSpecific_Error_10mm'],
                'Max_Error': metrics['TargetSpecific_Max_Error'],
                'Outliers_20mm': metrics['TargetSpecific_Outliers_20mm'],
                'Std_Error': metrics['TargetSpecific_Std_Error'],
                'Strategy': metrics['Strategy'],
                'Improvement': metrics['Improvement'],
                'Best_Single_Model': metrics['Best_Single_Model']
            }
            rows.append(ensemble_row)
            
            # Best single model row
            single_row = {
                'Target': target_name,
                'Model': 'Best_Single_Model',
                'MAE': metrics['BestSingle_MAE'],
                'RMSE': metrics['BestSingle_RMSE'],
                'R2': metrics['BestSingle_R2'],
                'Median_AE': metrics['BestSingle_Median_AE'],
                'Error_5mm': metrics['BestSingle_Error_5mm'],
                'Error_10mm': metrics['BestSingle_Error_10mm'],
                'Max_Error': metrics['BestSingle_Max_Error'],
                'Outliers_20mm': metrics['BestSingle_Outliers_20mm'],
                'Std_Error': metrics['BestSingle_Std_Error'],
                'Strategy': 'Single Best Model',
                'Improvement': 0.0,  # Reference point
                'Best_Single_Model': metrics['Best_Single_Model']
            }
            rows.append(single_row)
        
        results_df = pd.DataFrame(rows)
        results_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

def main():
    """Main training pipeline for target-specific ensemble models"""
    trainer = TargetSpecificEnsembleTrainer()
    
    # Load and prepare data
    df = trainer.load_and_prepare_data()
    if df is None:
        print("Failed to load data")
        return
    
    # Prepare training data
    X_train, X_test, y_train, y_test = trainer.prepare_training_data(df)
    
    # Train and evaluate target-specific ensembles
    results = trainer.train_and_evaluate_target_specific_ensembles(X_train, X_test, y_train, y_test)
    
    # Generate report
    trainer.generate_report()
    
    # Save results
    trainer.save_results()
    
    print(f"\nTarget-specific ensemble models training completed successfully!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Ridge Regression Model Training for Kidney Displacement Prediction
Regularized linear model with hyperparameter tuning
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import sys
import warnings
from pathlib import Path

warnings.warn(
    "train_ridge.py is legacy; use models/phase1/adaptive_ensemble.py. "
    "See: python scripts/run_phase1_pipeline.py info",
    DeprecationWarning,
    stacklevel=1,
)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.phase1_schema import BASE_FEATURES, TARGET_NAMES

class RidgeTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_selector = None
        self.feature_names = []
        self.target_names = []
        self.results = {}

        self.required_features = list(BASE_FEATURES)
        self.target_columns = list(TARGET_NAMES)
        
    def load_and_prepare_data(self):
        """Load and prepare datasets for training"""
        print("Loading datasets...")
        
        # Load datasets
        vybor_df = pd.read_csv('data/vybor_unified_features.csv')
        
        print(f"Vybor dataset: {len(vybor_df)} cases")
        
        # Process datasets
        combined_data = []
        
        # KiTS19 proxy deltas are not paired clinical labels — Vybor only.
        if 'kidney_left_delta_x' in vybor_df.columns:
            vybor_processed = self.process_dataset(vybor_df, 'vybor')
            if vybor_processed is not None:
                combined_data.append(vybor_processed)
        
        if combined_data:
            final_df = pd.concat(combined_data, ignore_index=True)

            for c in self.required_features:
                if c not in final_df.columns:
                    final_df[c] = np.nan

            feature_cols = list(self.required_features)
            target_cols = [c for c in self.target_columns if c in final_df.columns]
            final_df = final_df.dropna(subset=target_cols)
            final_df[feature_cols] = self.imputer.fit_transform(final_df[feature_cols])

            print(f"Combined dataset: {len(final_df)} cases")
            return final_df
        else:
            return None
    
    def process_dataset(self, df, dataset_name):
        """Process dataset to extract features and targets"""
        from src.features.phase1_schema import normalize_dataframe

        df = normalize_dataframe(df)
        available_targets = [col for col in self.target_columns if col in df.columns]

        if len(available_targets) == 0:
            print(f"No target variables found in {dataset_name} dataset")
            return None

        id_cols = [c for c in ('case_id',) if c in df.columns]
        cols = id_cols + list(self.required_features) + available_targets
        processed_df = df[cols].copy()
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
        
        # Scale features (important for Ridge regularization)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.feature_names = feature_cols
        self.target_names = target_cols
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def feature_selection(self, X_train, y_train, feature_names):
        """Perform feature selection using multiple methods"""
        print("\nPerforming feature selection...")
        
        # Method 1: SelectKBest with f_regression
        selector_kbest = SelectKBest(score_func=f_regression, k='all')
        selector_kbest.fit(X_train, y_train)
        kbest_scores = selector_kbest.scores_
        
        # Method 2: Random Forest importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_importance = rf.feature_importances_
        
        # Method 3: RFE with Ridge
        ridge = Ridge(random_state=42)
        rfe = RFE(estimator=ridge, n_features_to_select=min(20, len(feature_names)))
        rfe.fit(X_train, y_train)
        rfe_ranking = rfe.ranking_
        
        # Combine scores (normalized)
        kbest_norm = (kbest_scores - kbest_scores.min()) / (kbest_scores.max() - kbest_scores.min())
        rf_norm = (rf_importance - rf_importance.min()) / (rf_importance.max() - rf_importance.min())
        rfe_norm = (len(rfe_ranking) - rfe_ranking) / (len(rfe_ranking) - 1)
        
        # Weighted combination
        combined_scores = 0.4 * kbest_norm + 0.4 * rf_norm + 0.2 * rfe_norm
        
        # Select top features
        n_features = min(25, len(feature_names))  # Select top 25 features
        top_indices = np.argsort(combined_scores)[-n_features:]
        selected_features = [feature_names[i] for i in top_indices]
        
        print(f"Selected {len(selected_features)} features out of {len(feature_names)}")
        
        # Create feature selector
        self.feature_selector = SelectKBest(score_func=f_regression, k=n_features)
        self.feature_selector.fit(X_train, y_train)
        
        return selected_features
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform optimized hyperparameter tuning for Ridge"""
        print("\nPerforming optimized hyperparameter tuning for Ridge...")
        
        # Expanded parameter space
        param_distributions = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'saga'],
            'fit_intercept': [True, False],
            'max_iter': [1000, 2000, 5000],
            'tol': [1e-4, 1e-3, 1e-2],
            'positive': [True, False]
        }
        
        best_models = {}
        
        for i, target_name in enumerate(self.target_names):
            print(f"\nOptimizing for {target_name}...")
            
            y_train_target = y_train[:, i]
            
            # Feature selection for this target
            selected_features = self.feature_selection(X_train, y_train_target, self.feature_names)
            
            # Select features
            X_train_selected = self.feature_selector.transform(X_train)
            
            ridge = Ridge(random_state=42)
            
            random_search = RandomizedSearchCV(
                estimator=ridge,
                param_distributions=param_distributions,
                n_iter=100,  # Number of parameter settings sampled
                cv=5,
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                random_state=42,
                verbose=0
            )
            
            random_search.fit(X_train_selected, y_train_target)
            
            best_models[target_name] = {
                'model': random_search.best_estimator_,
                'selected_features': selected_features,
                'feature_selector': self.feature_selector
            }
            
            print(f"  Best params: {random_search.best_params_}")
            print(f"  Best CV MAE: {-random_search.best_score_:.3f}")
            print(f"  Features selected: {len(selected_features)}")
        
        return best_models
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, use_tuning=True):
        """Train Ridge models and evaluate performance"""
        if use_tuning:
            models = self.hyperparameter_tuning(X_train, y_train)
        else:
            models = {}
            for i, target_name in enumerate(self.target_names):
                y_train_target = y_train[:, i]
                selected_features = self.feature_selection(X_train, y_train_target, self.feature_names)
                models[target_name] = {
                    'model': Ridge(alpha=1.0, random_state=42),
                    'selected_features': selected_features,
                    'feature_selector': self.feature_selector,
                }
        
        results = {}
        
        print(f"\nTraining and evaluating optimized Ridge models...")
        
        for i, target_name in enumerate(self.target_names):
            print(f"\n{target_name}:")
            print("-" * 50)
            
            model_info = models[target_name]
            model = model_info['model']
            feature_selector = model_info['feature_selector']
            selected_features = model_info['selected_features']
            
            y_train_target = y_train[:, i]
            y_test_target = y_test[:, i]
            
            # Select features
            X_train_selected = feature_selector.transform(X_train)
            X_test_selected = feature_selector.transform(X_test)
            
            # Train model (already trained if use_tuning=True)
            if not use_tuning:
                model.fit(X_train_selected, y_train_target)
            
            # Predictions
            y_pred = model.predict(X_test_selected)
            
            # Metrics
            mae = mean_absolute_error(y_test_target, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test_target, y_pred))
            r2 = r2_score(y_test_target, y_pred)
            median_ae = np.median(np.abs(y_test_target - y_pred))
            
            # Clinical metrics
            error_5mm = np.mean(np.abs(y_test_target - y_pred) < 5) * 100
            error_10mm = np.mean(np.abs(y_test_target - y_pred) < 10) * 100
            max_error = np.max(np.abs(y_test_target - y_pred))
            outliers = np.sum(np.abs(y_test_target - y_pred) > 20)
            std_error = np.std(y_test_target - y_pred)
            
            cv_scores = cross_val_score(
                model,
                X_train_selected,
                y_train_target,
                cv=5,
                scoring='neg_mean_absolute_error',
            )
            cv_mae = -cv_scores.mean()
            cv_std = cv_scores.std()

            if hasattr(model, 'coef_'):
                coef = model.coef_
                nonzero_coef = int(np.sum(coef != 0))
                top_features = sorted(
                    zip(selected_features, np.abs(coef)),
                    key=lambda x: x[1],
                    reverse=True,
                )[:5]
            else:
                nonzero_coef = 0
                top_features = []

            results[target_name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'Median_AE': median_ae,
                'Error_5mm': error_5mm,
                'Error_10mm': error_10mm,
                'Max_Error': max_error,
                'Outliers_20mm': outliers,
                'Std_Error': std_error,
                'CV_MAE': cv_mae,
                'CV_Std': cv_std,
                'Model': model,
                'Alpha': model.alpha,
                'Solver': model.solver,
                'Selected_Features': len(selected_features),
                'NonZero_Coefficients': nonzero_coef,
                'Top_Features': top_features,
            }

            print(f"  Test MAE: {mae:.3f} mm")
            print(f"  Test RMSE: {rmse:.3f} mm")
            print(f"  R2: {r2:.3f}")
            print(f"  Median AE: {median_ae:.3f} mm")
            print(f"  <5mm accuracy: {error_5mm:.1f}%")
            print(f"  <10mm accuracy: {error_10mm:.1f}%")
            print(f"  Max Error: {max_error:.3f} mm")
            print(f"  Outliers >20mm: {outliers}")
            print(f"  CV MAE: {cv_mae:.3f} +/- {cv_std:.3f}")
            print(f"  Best Alpha: {model.alpha:.4f}")
            print(f"  Solver: {model.solver}")
            print(f"  Features selected: {len(selected_features)}")
            print(f"  Non-zero coefficients: {nonzero_coef}")

            if top_features:
                print(f"  Top 5 features:")
                for feat, coef_val in top_features:
                    print(f"    {feat}: {coef_val:.3f}")

        self.results = results
        return results
    
    def generate_report(self):
        """Generate comprehensive training report"""
        print("\n" + "="*80)
        print("OPTIMIZED RIDGE REGRESSION MODEL - DETAILED TRAINING REPORT")
        print("="*80)
        
        print(f"\nDataset Summary:")
        print(f"- Features used: {len(self.feature_names)}")
        print(f"- Target variables: {len(self.target_names)}")
        
        print(f"\nOverall Performance Summary:")
        print("-" * 50)
        
        # Calculate average metrics
        avg_mae = np.mean([r['MAE'] for r in self.results.values()])
        avg_rmse = np.mean([r['RMSE'] for r in self.results.values()])
        avg_r2 = np.mean([r['R2'] for r in self.results.values()])
        avg_5mm = np.mean([r['Error_5mm'] for r in self.results.values()])
        avg_10mm = np.mean([r['Error_10mm'] for r in self.results.values()])
        avg_features = np.mean([r['Selected_Features'] for r in self.results.values()])
        avg_nonzero = np.mean([r['NonZero_Coefficients'] for r in self.results.values()])
        
        print(f"Average MAE: {avg_mae:.3f} mm")
        print(f"Average RMSE: {avg_rmse:.3f} mm")
        print(f"Average R²: {avg_r2:.3f}")
        print(f"Average <5mm accuracy: {avg_5mm:.1f}%")
        print(f"Average <10mm accuracy: {avg_10mm:.1f}%")
        print(f"Average features selected: {avg_features:.0f}")
        print(f"Average non-zero coefficients: {avg_nonzero:.0f}")
        
        print(f"\nFeature Selection Analysis:")
        print("-" * 50)
        
        avg_selected = np.mean([r['Selected_Features'] for r in self.results.values()])
        avg_nonzero = np.mean([r['NonZero_Coefficients'] for r in self.results.values()])
        
        print(f"Average features selected: {avg_selected:.0f}/{len(self.feature_names)}")
        print(f"Average non-zero coefficients: {avg_nonzero:.0f}")
        
        print(f"\nDetailed Results by Target:")
        print("-" * 50)
        
        for target_name, metrics in self.results.items():
            print(f"\n{target_name}:")
            print(f"  MAE: {metrics['MAE']:.3f} mm (CV: {metrics['CV_MAE']:.3f} ± {metrics['CV_Std']:.3f})")
            print(f"  RMSE: {metrics['RMSE']:.3f} mm")
            print(f"  R²: {metrics['R2']:.3f}")
            print(f"  Clinical accuracy: {metrics['Error_5mm']:.1f}% <5mm, {metrics['Error_10mm']:.1f}% <10mm")
            print(f"  Alpha: {metrics['Alpha']:.3f}, Solver: {metrics['Solver']}")
            print(f"  Features: {metrics['Selected_Features']} selected, {metrics['NonZero_Coefficients']} non-zero")
    
    def save_results(self, filename='ridge_results.csv'):
        """Save results to CSV"""
        rows = []
        for target_name, metrics in self.results.items():
            row = {
                'Target': target_name,
                'Model': 'Ridge_Optimized',
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'R2': metrics['R2'],
                'Median_AE': metrics['Median_AE'],
                'Error_5mm': metrics['Error_5mm'],
                'Error_10mm': metrics['Error_10mm'],
                'Max_Error': metrics['Max_Error'],
                'Outliers_20mm': metrics['Outliers_20mm'],
                'Std_Error': metrics['Std_Error'],
                'CV_MAE': metrics['CV_MAE'],
                'CV_Std': metrics['CV_Std'],
                'Alpha': metrics['Alpha'],
                'Solver': metrics['Solver'],
                'Selected_Features': metrics['Selected_Features'],
                'NonZero_Coefficients': metrics['NonZero_Coefficients']
            }
            rows.append(row)
        
        results_df = pd.DataFrame(rows)
        results_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

def main():
    """Main training pipeline for Ridge regression"""
    trainer = RidgeTrainer()
    
    # Load and prepare data
    df = trainer.load_and_prepare_data()
    if df is None:
        print("Failed to load data")
        return
    
    # Prepare training data
    X_train, X_test, y_train, y_test = trainer.prepare_training_data(df)
    
    # Train and evaluate models
    results = trainer.train_and_evaluate(X_train, X_test, y_train, y_test, use_tuning=True)
    
    # Generate report
    trainer.generate_report()
    
    # Save results
    trainer.save_results()
    
    print(f"\nOptimized Ridge regression training completed successfully!")

if __name__ == "__main__":
    main()

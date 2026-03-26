#!/usr/bin/env python3
"""
Uncertainty Quantification Predictor for Kidney Displacement
Phase 3: Bayesian approaches with confidence estimation
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class BayesianLinear(nn.Module):
    """Bayesian linear layer with uncertainty estimation"""
    
    def __init__(self, in_features, out_features, prior_std=1.0):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameters (mean and log variance)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Bias parameters (mean and log variance)
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_logvar = nn.Parameter(torch.Tensor(out_features))
        
        # Prior standard deviation
        self.prior_std = prior_std
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_normal_(self.weight_mu)
        nn.init.constant_(self.weight_logvar, -3)  # Start with small uncertainty
        nn.init.normal_(self.bias_mu, 0, 0.1)
        nn.init.constant_(self.bias_logvar, -3)
    
    def forward(self, x):
        if self.training:
            # Sample weights and biases during training
            weight_std = torch.exp(0.5 * self.weight_logvar)
            weight_sample = self.weight_mu + weight_std * torch.randn_like(self.weight_mu)
            
            bias_std = torch.exp(0.5 * self.bias_logvar)
            bias_sample = self.bias_mu + bias_std * torch.randn_like(self.bias_mu)
        else:
            # Use mean weights and biases during evaluation
            weight_sample = self.weight_mu
            bias_sample = self.bias_mu
        
        return nn.functional.linear(x, weight_sample, bias_sample)
    
    def kl_divergence(self):
        """Calculate KL divergence from prior"""
        # KL divergence for weights
        weight_kl = 0.5 * torch.sum(
            self.prior_std**-2 * (self.weight_mu**2 + torch.exp(self.weight_logvar)) - 1 + self.weight_logvar
        )
        
        # KL divergence for bias
        bias_kl = 0.5 * torch.sum(
            self.prior_std**-2 * (self.bias_mu**2 + torch.exp(self.bias_logvar)) - 1 + self.bias_logvar
        )
        
        return weight_kl + bias_kl

class MonteCarloDropout(nn.Module):
    """Monte Carlo dropout for uncertainty estimation"""
    
    def __init__(self, p=0.2):
        super(MonteCarloDropout, self).__init__()
        self.p = p
    
    def forward(self, x):
        return nn.functional.dropout(x, p=self.p, training=True)

class UncertaintyNet(nn.Module):
    """Neural network with uncertainty quantification"""
    
    def __init__(self, input_dim=30, hidden_dims=[256, 128, 64], output_dim=6, dropout=0.2):
        super(UncertaintyNet, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Feature extraction layers with Bayesian linear layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                # First layer is Bayesian
                layers.append(BayesianLinear(prev_dim, hidden_dim))
            else:
                layers.append(nn.Linear(prev_dim, hidden_dim))
            
            layers.extend([
                nn.ReLU(),
                MonteCarloDropout(dropout) if i < len(hidden_dims) - 1 else nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output layers with uncertainty
        self.output_mu = BayesianLinear(hidden_dims[-1], output_dim)
        self.output_logvar = nn.Linear(hidden_dims[-1], output_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Output mean and log variance
        output_mu = self.output_mu(features)
        output_logvar = self.output_logvar(features)
        
        return output_mu, output_logvar
    
    def kl_divergence(self):
        """Calculate total KL divergence"""
        kl_div = 0
        for module in self.modules():
            if isinstance(module, BayesianLinear):
                kl_div += module.kl_divergence()
        return kl_div

class UncertaintyDataset(Dataset):
    """PyTorch Dataset for uncertainty quantification"""
    
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class UncertaintyLoss(nn.Module):
    """Custom loss function for uncertainty quantification"""
    
    def __init__(self, beta=1.0):
        super(UncertaintyLoss, self).__init__()
        self.beta = beta  # Weight for KL divergence term
    
    def forward(self, pred_mu, pred_logvar, targets, kl_divergence):
        """
        Calculate loss with uncertainty estimation
        
        Args:
            pred_mu: Predicted mean
            pred_logvar: Predicted log variance
            targets: True values
            kl_divergence: KL divergence from prior
        """
        # Negative log likelihood with heteroscedastic uncertainty
        pred_var = torch.exp(pred_logvar)
        nll = 0.5 * torch.sum(
            torch.log(pred_var) + (targets - pred_mu)**2 / pred_var
        )
        
        # Add KL divergence for regularization
        total_loss = nll + self.beta * kl_divergence
        
        return total_loss

class UncertaintyQuantificationPredictor:
    """Predictor with uncertainty quantification"""
    
    def __init__(self, input_dim=30, device='cpu'):
        self.input_dim = input_dim
        self.device = device
        
        # Initialize model
        self.model = UncertaintyNet(input_dim=input_dim, hidden_dims=[256, 128, 64], output_dim=6).to(device)
        
        # Custom loss function
        self.criterion = UncertaintyLoss(beta=1.0)
        
        # Data preprocessing
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        
        # Target names
        self.target_names = [
            'kidney_left_delta_x', 'kidney_left_delta_y', 'kidney_left_delta_z',
            'kidney_right_delta_x', 'kidney_right_delta_y', 'kidney_right_delta_z'
        ]
    
    def prepare_data(self, df):
        """Prepare data for uncertainty quantification"""
        print("Preparing data for uncertainty quantification...")
        
        # Feature columns
        feature_columns = [
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
            'body_com_x', 'body_com_y', 'body_com_z'
        ]
        
        # Add patient_position_encoded if available
        if 'patient_position_encoded' in df.columns:
            feature_columns.append('patient_position_encoded')
        else:
            # Create patient_position_encoded from position data
            position_col = 'patient_position' if 'patient_position' in df.columns else 'scan_position'
            if position_col in df.columns:
                pos = df[position_col].fillna('supine').astype(str).str.lower()
                df['patient_position_encoded'] = (~pos.str.contains('sup')).astype(int)
                feature_columns.append('patient_position_encoded')
            else:
                df['patient_position_encoded'] = 0
                feature_columns.append('patient_position_encoded')
        
        # Target columns
        target_columns = self.target_names
        
        # Extract features and targets
        X = df[feature_columns].values
        y = df[target_columns].values
        
        # Handle missing values
        X = self.imputer.fit_transform(X)
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, train_loader, val_loader, epochs=100, lr=0.001):
        """Train the uncertainty quantification model"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 20
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                pred_mu, pred_logvar = self.model(batch_features)
                
                # Calculate KL divergence
                kl_div = self.model.kl_divergence()
                
                # Calculate loss
                loss = self.criterion(pred_mu, pred_logvar, batch_targets, kl_div)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    pred_mu, pred_logvar = self.model(batch_features)
                    kl_div = self.model.kl_divergence()
                    loss = self.criterion(pred_mu, pred_logvar, batch_targets, kl_div)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Load best model
        self.model.load_state_dict(best_model_state)
        
        return train_losses, val_losses
    
    def train(self, X_train, X_test, y_train, y_test, epochs=100, batch_size=32):
        """Train the uncertainty quantification model"""
        print("Training uncertainty quantification model...")
        
        # Create datasets and dataloaders
        train_dataset = UncertaintyDataset(X_train, y_train)
        test_dataset = UncertaintyDataset(X_test, y_test)
        
        # Split training data for validation
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Train model
        train_losses, val_losses = self.train_model(train_loader, val_loader, epochs)
        
        return train_losses, val_losses
    
    def predict_with_uncertainty(self, X, n_samples=100):
        """Make predictions with uncertainty estimation using Monte Carlo dropout"""
        self.model.train()  # Enable dropout for uncertainty estimation
        
        all_predictions = []
        all_uncertainties = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                X_tensor = torch.FloatTensor(X).to(self.device)
                pred_mu, pred_logvar = self.model(X_tensor)
                
                all_predictions.append(pred_mu.cpu().numpy())
                all_uncertainties.append(torch.exp(pred_logvar).cpu().numpy())
        
        # Calculate mean and standard deviation across samples
        predictions_mean = np.mean(all_predictions, axis=0)
        predictions_std = np.std(all_predictions, axis=0)
        
        # Combine model uncertainty with Monte Carlo uncertainty
        model_uncertainty = np.mean(all_uncertainties, axis=0)
        total_uncertainty = np.sqrt(predictions_std**2 + model_uncertainty)
        
        self.model.eval()  # Return to evaluation mode
        
        return predictions_mean, total_uncertainty, predictions_std
    
    def evaluate_with_uncertainty(self, X_test, y_test, n_samples=100):
        """Evaluate model with uncertainty quantification"""
        print("Evaluating uncertainty quantification model...")
        
        predictions_mean, total_uncertainty, predictions_std = self.predict_with_uncertainty(X_test, n_samples)
        
        # Calculate metrics for each target
        results = {}
        uncertainty_results = {}
        
        for i, target_name in enumerate(self.target_names):
            y_true = y_test[:, i]
            y_pred_mean = predictions_mean[:, i]
            y_uncertainty = total_uncertainty[:, i]
            y_pred_std = predictions_std[:, i]
            
            # Basic metrics
            mae = mean_absolute_error(y_true, y_pred_mean)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred_mean))
            r2 = r2_score(y_true, y_pred_mean)
            median_ae = np.median(np.abs(y_true - y_pred_mean))
            
            # Clinical metrics
            error_5mm = np.mean(np.abs(y_true - y_pred_mean) < 5) * 100
            error_10mm = np.mean(np.abs(y_true - y_pred_mean) < 10) * 100
            max_error = np.max(np.abs(y_true - y_pred_mean))
            outliers = np.sum(np.abs(y_true - y_pred_mean) > 20)
            std_error = np.std(y_true - y_pred_mean)
            
            # Uncertainty metrics
            avg_uncertainty = np.mean(y_uncertainty)
            uncertainty_calibration = self._calculate_uncertainty_calibration(y_true, y_pred_mean, y_uncertainty)
            coverage_95 = np.mean((y_true >= y_pred_mean - 1.96*y_uncertainty) & 
                                 (y_true <= y_pred_mean + 1.96*y_uncertainty)) * 100
            
            results[target_name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'Median_AE': median_ae,
                'Error_5mm': error_5mm,
                'Error_10mm': error_10mm,
                'Max_Error': max_error,
                'Outliers_20mm': outliers,
                'Std_Error': std_error
            }
            
            uncertainty_results[target_name] = {
                'Average_Uncertainty': avg_uncertainty,
                'Uncertainty_Calibration': uncertainty_calibration,
                'Coverage_95': coverage_95,
                'MC_Std': np.mean(y_pred_std)
            }
        
        # Calculate overall metrics
        overall_metrics = {
            'Average_MAE': np.mean([r['MAE'] for r in results.values()]),
            'Average_RMSE': np.mean([r['RMSE'] for r in results.values()]),
            'Average_R2': np.mean([r['R2'] for r in results.values()]),
            'Average_Error_5mm': np.mean([r['Error_5mm'] for r in results.values()]),
            'Average_Error_10mm': np.mean([r['Error_10mm'] for r in results.values()])
        }
        
        overall_uncertainty = {
            'Average_Uncertainty': np.mean([u['Average_Uncertainty'] for u in uncertainty_results.values()]),
            'Average_Calibration': np.mean([u['Uncertainty_Calibration'] for u in uncertainty_results.values()]),
            'Average_Coverage_95': np.mean([u['Coverage_95'] for u in uncertainty_results.values()])
        }
        
        return results, overall_metrics, uncertainty_results
    
    def _calculate_uncertainty_calibration(self, y_true, y_pred, y_uncertainty):
        """Calculate uncertainty calibration score"""
        # Normalize residuals by predicted uncertainty
        normalized_residuals = np.abs(y_true - y_pred) / (y_uncertainty + 1e-8)
        
        # Ideal calibration should have mean of 1.0 (for 1-sigma)
        calibration_score = np.mean(normalized_residuals)
        
        return calibration_score
    
    def analyze_uncertainty_patterns(self, X_test, y_test, n_samples=100):
        """Analyze uncertainty patterns and relationships"""
        print("\nAnalyzing uncertainty patterns...")
        
        predictions_mean, total_uncertainty, predictions_std = self.predict_with_uncertainty(X_test, n_samples)
        
        # Calculate prediction errors
        prediction_errors = np.abs(y_test - predictions_mean)
        
        # Analyze relationship between uncertainty and error
        uncertainty_error_correlation = []
        
        for i in range(len(self.target_names)):
            correlation = np.corrcoef(total_uncertainty[:, i], prediction_errors[:, i])[0, 1]
            uncertainty_error_correlation.append((self.target_names[i], correlation))
        
        # Sort by correlation
        uncertainty_error_correlation.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print("Uncertainty-Error Correlations:")
        for target_name, correlation in uncertainty_error_correlation:
            print(f"  {target_name}: {correlation:.3f}")
        
        # Find high uncertainty cases
        avg_uncertainty = np.mean(total_uncertainty, axis=1)
        high_uncertainty_threshold = np.percentile(avg_uncertainty, 90)
        high_uncertainty_indices = np.where(avg_uncertainty > high_uncertainty_threshold)[0]
        
        print(f"\nHigh uncertainty cases (top 10%): {len(high_uncertainty_indices)} samples")
        if len(high_uncertainty_indices) > 0:
            high_uncertainty_errors = prediction_errors[high_uncertainty_indices]
            print(f"  Average error in high uncertainty cases: {np.mean(high_uncertainty_errors):.3f} mm")
            print(f"  Max error in high uncertainty cases: {np.max(high_uncertainty_errors):.3f} mm")
        
        return uncertainty_error_correlation, high_uncertainty_indices
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'criterion_state_dict': self.criterion.state_dict(),
            'scaler': self.scaler,
            'imputer': self.imputer,
            'input_dim': self.input_dim,
            'target_names': self.target_names
        }
        
        torch.save(model_state, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        model_state = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(model_state['model_state_dict'])
        self.criterion.load_state_dict(model_state['criterion_state_dict'])
        self.scaler = model_state['scaler']
        self.imputer = model_state['imputer']
        
        print(f"Model loaded from {filepath}")

def main():
    """Main training pipeline for uncertainty quantification"""
    print("="*80)
    print("UNCERTAINTY QUANTIFICATION PREDICTOR - PHASE 3")
    print("="*80)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print("Loading datasets...")
    vybor_df = pd.read_csv('data/vybor_unified_features.csv')
    kits19_df = pd.read_csv('data/kits19_medical_grade_features.csv')
    
    # Combine datasets
    combined_data = []
    
    if 'kidney_left_delta_x' in vybor_df.columns:
        combined_data.append(vybor_df)
    
    if 'kidney_left_delta_x' in kits19_df.columns:
        combined_data.append(kits19_df)
    
    if combined_data:
        df = pd.concat(combined_data, ignore_index=True)
        print(f"Combined dataset: {len(df)} cases")
        
        # Initialize predictor
        predictor = UncertaintyQuantificationPredictor(input_dim=30, device=device)
        
        # Prepare data
        X_train, X_test, y_train, y_test = predictor.prepare_data(df)
        
        # Train model
        train_losses, val_losses = predictor.train(X_train, X_test, y_train, y_test, epochs=100, batch_size=16)
        
        # Evaluate model with uncertainty
        results, overall_metrics, uncertainty_results = predictor.evaluate_with_uncertainty(X_test, y_test)
        
        # Analyze uncertainty patterns
        uncertainty_correlations, high_uncertainty_indices = predictor.analyze_uncertainty_patterns(X_test, y_test)
        
        # Generate report
        print("\n" + "="*80)
        print("UNCERTAINTY QUANTIFICATION RESULTS")
        print("="*80)
        
        print(f"\nOverall Performance:")
        print(f"  Average MAE: {overall_metrics['Average_MAE']:.3f} mm")
        print(f"  Average RMSE: {overall_metrics['Average_RMSE']:.3f} mm")
        print(f"  Average R2: {overall_metrics['Average_R2']:.3f}")
        print(f"  Average <5mm accuracy: {overall_metrics['Average_Error_5mm']:.1f}%")
        print(f"  Average <10mm accuracy: {overall_metrics['Average_Error_10mm']:.1f}%")
        
        print(f"\nUncertainty Metrics:")
        print(f"  Average Uncertainty: {overall_uncertainty['Average_Uncertainty']:.3f} mm")
        print(f"  Average Calibration: {overall_uncertainty['Average_Calibration']:.3f}")
        print(f"  Average 95% Coverage: {overall_uncertainty['Average_Coverage_95']:.1f}%")
        
        print(f"\nPer-Target Performance:")
        for target_name, metrics in results.items():
            uncertainty_metrics = uncertainty_results[target_name]
            print(f"  {target_name}:")
            print(f"    MAE: {metrics['MAE']:.3f} mm")
            print(f"    R²: {metrics['R2']:.3f}")
            print(f"    <5mm accuracy: {metrics['Error_5mm']:.1f}%")
            print(f"    Uncertainty: {uncertainty_metrics['Average_Uncertainty']:.3f} mm")
            print(f"    95% Coverage: {uncertainty_metrics['Coverage_95']:.1f}%")
            print(f"    MC Std: {uncertainty_metrics['MC_Std']:.3f} mm")
        
        # Save results
        results_df = pd.DataFrame([
            {'Target': target, **metrics} for target, metrics in results.items()
        ])
        results_df.to_csv('uncertainty_quantification_results.csv', index=False)
        
        # Save uncertainty results
        uncertainty_df = pd.DataFrame([
            {'Target': target, **uncertainty_metrics} for target, uncertainty_metrics in uncertainty_results.items()
        ])
        uncertainty_df.to_csv('uncertainty_metrics_results.csv', index=False)
        
        # Save model
        predictor.save_model('uncertainty_quantification_model.pth')
        
        print(f"\n✅ Uncertainty quantification model training completed!")
        print(f"Results saved to uncertainty_quantification_results.csv")
        print(f"Uncertainty metrics saved to uncertainty_metrics_results.csv")
        print(f"Model saved to uncertainty_quantification_model.pth")
        
    else:
        print("No valid datasets found!")

if __name__ == "__main__":
    main()

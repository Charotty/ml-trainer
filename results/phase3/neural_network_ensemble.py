#!/usr/bin/env python3
"""
Neural Network Ensemble for Kidney Displacement Prediction
Phase 3: Advanced deep learning approaches with attention mechanisms
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

class KidneyDisplacementDataset(Dataset):
    """PyTorch Dataset for kidney displacement prediction"""
    
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class AttentionMechanism(nn.Module):
    """Multi-head attention mechanism for feature importance"""
    
    def __init__(self, input_dim, num_heads=4, dropout=0.1):
        super(AttentionMechanism, self).__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.head_dim = input_dim // num_heads
        
        # Ensure input_dim is divisible by num_heads
        if input_dim % num_heads != 0:
            # Adjust to make it divisible
            self.head_dim = 32  # Fixed head dimension
            self.adjusted_dim = self.head_dim * num_heads
        else:
            self.adjusted_dim = input_dim
        
        self.query = nn.Linear(input_dim, self.adjusted_dim)
        self.key = nn.Linear(input_dim, self.adjusted_dim)
        self.value = nn.Linear(input_dim, self.adjusted_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.adjusted_dim)
        
        # Output projection to match original input_dim
        self.output_proj = nn.Linear(self.adjusted_dim, input_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Linear transformations
        Q = self.query(x).view(batch_size, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, self.num_heads, self.head_dim)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        attended = attended.view(batch_size, self.adjusted_dim)
        
        # Handle dimension mismatch for residual connection
        if self.adjusted_dim != x.size(1):
            # Project x to match attended dimensions
            x_proj = nn.Linear(x.size(1), self.adjusted_dim).to(x.device)(x)
            output = self.layer_norm(attended + x_proj)
        else:
            output = self.layer_norm(attended + x)
        
        # Project back to original dimension
        output = self.output_proj(output)
        
        return output, attention_weights.mean(dim=1)  # Return mean attention weights

class KidneyDisplacementNet(nn.Module):
    """Advanced neural network with attention for kidney displacement prediction"""
    
    def __init__(self, input_dim=30, hidden_dims=[256, 128, 64], output_dim=6, dropout=0.2):
        super(KidneyDisplacementNet, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Feature extraction layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Attention mechanism
        self.attention = AttentionMechanism(hidden_dims[-1], num_heads=4, dropout=dropout)
        
        # Output layers
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dims[-1], 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1)
            ) for _ in range(output_dim)
        ])
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Apply attention
        attended_features, attention_weights = self.attention(features)
        
        # Multi-task output
        outputs = []
        for output_layer in self.output_layers:
            output = output_layer(attended_features)
            outputs.append(output.squeeze(-1))
        
        return torch.stack(outputs, dim=1), attention_weights

class NeuralNetworkEnsemble:
    """Ensemble of neural networks with different architectures"""
    
    def __init__(self, input_dim=30, output_dim=6, device='cpu'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        
        # Create ensemble of different architectures
        self.models = []
        self.model_configs = [
            {'hidden_dims': [256, 128, 64], 'dropout': 0.2},
            {'hidden_dims': [512, 256, 128, 64], 'dropout': 0.3},
            {'hidden_dims': [128, 64, 32], 'dropout': 0.1},
            {'hidden_dims': [384, 192, 96], 'dropout': 0.25},
            {'hidden_dims': [200, 100, 50], 'dropout': 0.15}
        ]
        
        for config in self.model_configs:
            model = KidneyDisplacementNet(
                input_dim=input_dim,
                hidden_dims=config['hidden_dims'],
                output_dim=output_dim,
                dropout=config['dropout']
            ).to(device)
            self.models.append(model)
        
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        
        # Target names
        self.target_names = [
            'kidney_left_delta_x', 'kidney_left_delta_y', 'kidney_left_delta_z',
            'kidney_right_delta_x', 'kidney_right_delta_y', 'kidney_right_delta_z'
        ]
    
    def prepare_data(self, df):
        """Prepare data for neural network training"""
        print("Preparing data for neural network training...")
        
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
    
    def train_model(self, model, train_loader, val_loader, epochs=100, lr=0.001):
        """Train a single neural network model"""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 20
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                outputs, _ = model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    outputs, _ = model(batch_features)
                    loss = criterion(outputs, batch_targets)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        return train_losses, val_losses
    
    def train_ensemble(self, X_train, X_test, y_train, y_test, epochs=100, batch_size=32):
        """Train the entire ensemble of models"""
        print("Training neural network ensemble...")
        
        # Create datasets and dataloaders
        train_dataset = KidneyDisplacementDataset(X_train, y_train)
        test_dataset = KidneyDisplacementDataset(X_test, y_test)
        
        # Split training data for validation
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Train each model in the ensemble
        ensemble_results = []
        
        for i, model in enumerate(self.models):
            print(f"\nTraining model {i+1}/{len(self.models)}...")
            print(f"Architecture: {self.model_configs[i]}")
            
            train_losses, val_losses = self.train_model(model, train_loader, val_loader, epochs)
            
            ensemble_results.append({
                'model': model,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'config': self.model_configs[i]
            })
        
        return ensemble_results
    
    def predict_ensemble(self, X):
        """Make predictions using the ensemble"""
        self.models[0].eval()
        
        all_predictions = []
        all_attention_weights = []
        
        with torch.no_grad():
            for model in self.models:
                model.eval()
                X_tensor = torch.FloatTensor(X).to(self.device)
                predictions, attention_weights = model(X_tensor)
                all_predictions.append(predictions.cpu().numpy())
                all_attention_weights.append(attention_weights.cpu().numpy())
        
        # Average predictions across ensemble
        ensemble_predictions = np.mean(all_predictions, axis=0)
        
        # Average attention weights
        ensemble_attention = np.mean(all_attention_weights, axis=0)
        
        return ensemble_predictions, ensemble_attention
    
    def evaluate_ensemble(self, X_test, y_test):
        """Evaluate ensemble performance"""
        print("Evaluating neural network ensemble...")
        
        predictions, attention_weights = self.predict_ensemble(X_test)
        
        # Calculate metrics for each target
        results = {}
        
        for i, target_name in enumerate(self.target_names):
            y_true = y_test[:, i]
            y_pred = predictions[:, i]
            
            # Basic metrics
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            median_ae = np.median(np.abs(y_true - y_pred))
            
            # Clinical metrics
            error_5mm = np.mean(np.abs(y_true - y_pred) < 5) * 100
            error_10mm = np.mean(np.abs(y_true - y_pred) < 10) * 100
            max_error = np.max(np.abs(y_true - y_pred))
            outliers = np.sum(np.abs(y_true - y_pred) > 20)
            std_error = np.std(y_true - y_pred)
            
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
        
        # Calculate overall metrics
        overall_metrics = {
            'Average_MAE': np.mean([r['MAE'] for r in results.values()]),
            'Average_RMSE': np.mean([r['RMSE'] for r in results.values()]),
            'Average_R2': np.mean([r['R2'] for r in results.values()]),
            'Average_Error_5mm': np.mean([r['Error_5mm'] for r in results.values()]),
            'Average_Error_10mm': np.mean([r['Error_10mm'] for r in results.values()])
        }
        
        return results, overall_metrics, attention_weights
    
    def analyze_attention_weights(self, attention_weights, feature_names):
        """Analyze attention weights to understand feature importance"""
        print("\nAnalyzing attention weights...")
        
        # Average attention weights across all samples
        avg_attention = np.mean(attention_weights, axis=0)
        
        # Sort features by attention weight
        feature_importance = list(zip(feature_names, avg_attention))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print("Top 10 most important features (by attention):")
        for i, (feature, importance) in enumerate(feature_importance[:10]):
            print(f"  {i+1}. {feature}: {importance:.4f}")
        
        return feature_importance
    
    def save_ensemble(self, filepath):
        """Save the trained ensemble"""
        ensemble_state = {
            'models': [model.state_dict() for model in self.models],
            'model_configs': self.model_configs,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim
        }
        
        torch.save(ensemble_state, filepath)
        print(f"Ensemble saved to {filepath}")
    
    def load_ensemble(self, filepath):
        """Load a trained ensemble"""
        ensemble_state = torch.load(filepath, map_location=self.device)
        
        # Load model states
        for i, (model, state_dict) in enumerate(zip(self.models, ensemble_state['models'])):
            model.load_state_dict(state_dict)
        
        self.scaler = ensemble_state['scaler']
        self.imputer = ensemble_state['imputer']
        
        print(f"Ensemble loaded from {filepath}")

def main():
    """Main training pipeline for neural network ensemble"""
    print("="*80)
    print("NEURAL NETWORK ENSEMBLE - PHASE 3")
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
        
        # Initialize ensemble
        ensemble = NeuralNetworkEnsemble(input_dim=30, output_dim=6, device=device)
        
        # Prepare data
        X_train, X_test, y_train, y_test = ensemble.prepare_data(df)
        
        # Train ensemble
        ensemble_results = ensemble.train_ensemble(X_train, X_test, y_train, y_test, epochs=100, batch_size=16)
        
        # Evaluate ensemble
        results, overall_metrics, attention_weights = ensemble.evaluate_ensemble(X_test, y_test)
        
        # Analyze attention weights
        feature_names = [
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
        
        feature_importance = ensemble.analyze_attention_weights(attention_weights, feature_names)
        
        # Generate report
        print("\n" + "="*80)
        print("NEURAL NETWORK ENSEMBLE RESULTS")
        print("="*80)
        
        print(f"\nOverall Performance:")
        print(f"  Average MAE: {overall_metrics['Average_MAE']:.3f} mm")
        print(f"  Average RMSE: {overall_metrics['Average_RMSE']:.3f} mm")
        print(f"  Average R²: {overall_metrics['Average_R2']:.3f}")
        print(f"  Average <5mm accuracy: {overall_metrics['Average_Error_5mm']:.1f}%")
        print(f"  Average <10mm accuracy: {overall_metrics['Average_Error_10mm']:.1f}%")
        
        print(f"\nPer-Target Performance:")
        for target_name, metrics in results.items():
            print(f"  {target_name}:")
            print(f"    MAE: {metrics['MAE']:.3f} mm")
            print(f"    R²: {metrics['R2']:.3f}")
            print(f"    <5mm accuracy: {metrics['Error_5mm']:.1f}%")
        
        # Save results
        results_df = pd.DataFrame([
            {'Target': target, **metrics} for target, metrics in results.items()
        ])
        results_df.to_csv('neural_network_ensemble_results.csv', index=False)
        
        # Save ensemble
        ensemble.save_ensemble('neural_network_ensemble.pth')
        
        print(f"\n✅ Neural network ensemble training completed!")
        print(f"Results saved to neural_network_ensemble_results.csv")
        print(f"Model saved to neural_network_ensemble.pth")
        
    else:
        print("No valid datasets found!")

if __name__ == "__main__":
    main()

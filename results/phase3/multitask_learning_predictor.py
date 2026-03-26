#!/usr/bin/env python3
"""
Multitask Learning Predictor for Kidney Displacement
Phase 3: Hierarchical multitask learning with shared representations
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

class MultitaskDataset(Dataset):
    """PyTorch Dataset for multitask learning"""
    
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class SharedRepresentation(nn.Module):
    """Shared representation layer for multitask learning"""
    
    def __init__(self, input_dim, shared_dim=128, dropout=0.2):
        super(SharedRepresentation, self).__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Task-specific attention
        self.task_attention = nn.MultiheadAttention(
            embed_dim=shared_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(shared_dim)
    
    def forward(self, x):
        # Shared representation
        shared_features = self.shared_layers(x)
        
        # Add batch dimension for attention
        shared_features_expanded = shared_features.unsqueeze(1)
        
        # Self-attention
        attended, attention_weights = self.task_attention(
            shared_features_expanded,
            shared_features_expanded,
            shared_features_expanded
        )
        
        # Remove batch dimension and residual connection
        attended = attended.squeeze(1)
        output = self.layer_norm(attended + shared_features)
        
        return output, attention_weights.squeeze(1)

class TaskSpecificHead(nn.Module):
    """Task-specific head for individual prediction tasks"""
    
    def __init__(self, shared_dim, hidden_dims=[64, 32], dropout=0.1):
        super(TaskSpecificHead, self).__init__()
        
        layers = []
        prev_dim = shared_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.head = nn.Sequential(*layers)
    
    def forward(self, shared_features):
        return self.head(shared_features).squeeze(-1)

class HierarchicalMultitaskNet(nn.Module):
    """Hierarchical multitask network for kidney displacement prediction"""
    
    def __init__(self, input_dim=30, shared_dim=128, dropout=0.2):
        super(HierarchicalMultitaskNet, self).__init__()
        
        # Task groups based on anatomical relationships
        self.task_groups = {
            'left_kidney': ['kidney_left_delta_x', 'kidney_left_delta_y', 'kidney_left_delta_z'],
            'right_kidney': ['kidney_right_delta_x', 'kidney_right_delta_y', 'kidney_right_delta_z'],
            'x_axis': ['kidney_left_delta_x', 'kidney_right_delta_x'],
            'y_axis': ['kidney_left_delta_y', 'kidney_right_delta_y'],
            'z_axis': ['kidney_left_delta_z', 'kidney_right_delta_z']
        }
        
        # All target names
        self.target_names = [
            'kidney_left_delta_x', 'kidney_left_delta_y', 'kidney_left_delta_z',
            'kidney_right_delta_x', 'kidney_right_delta_y', 'kidney_right_delta_z'
        ]
        
        self.target_to_idx = {name: i for i, name in enumerate(self.target_names)}
        
        # Shared representation
        self.shared_representation = SharedRepresentation(input_dim, shared_dim, dropout)
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict({
            target: TaskSpecificHead(shared_dim, dropout=dropout)
            for target in self.target_names
        })
        
        # Group-level representations
        self.group_representations = nn.ModuleDict({
            group: nn.Sequential(
                nn.Linear(shared_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for group in self.task_groups.keys()
        })
        
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
    
    def forward(self, x, return_attention=False):
        # Shared representation
        shared_features, attention_weights = self.shared_representation(x)
        
        # Task-specific predictions
        predictions = {}
        
        for target_name, head in self.task_heads.items():
            predictions[target_name] = head(shared_features)
        
        # Stack predictions in the correct order
        output = torch.stack([predictions[target] for target in self.target_names], dim=1)
        
        if return_attention:
            return output, attention_weights
        else:
            return output

class MultitaskLoss(nn.Module):
    """Custom multitask loss with task uncertainty weighting"""
    
    def __init__(self, task_weights=None, learn_uncertainty=True):
        super(MultitaskLoss, self).__init__()
        
        self.target_names = [
            'kidney_left_delta_x', 'kidney_left_delta_y', 'kidney_left_delta_z',
            'kidney_right_delta_x', 'kidney_right_delta_y', 'kidney_right_delta_z'
        ]
        
        # Task-specific weights (based on difficulty)
        if task_weights is None:
            self.task_weights = nn.Parameter(torch.ones(len(self.target_names)))
        else:
            self.task_weights = nn.Parameter(torch.tensor(task_weights))
        
        # Learnable uncertainty parameters
        if learn_uncertainty:
            self.log_vars = nn.Parameter(torch.zeros(len(self.target_names)))
        else:
            self.log_vars = torch.zeros(len(self.target_names))
    
    def forward(self, predictions, targets):
        """
        Calculate multitask loss with uncertainty weighting
        
        Args:
            predictions: Model predictions (batch_size, num_tasks)
            targets: True values (batch_size, num_tasks)
        """
        mse_loss = nn.MSELoss(reduction='none')
        
        # Calculate task-specific losses
        task_losses = []
        for i in range(len(self.target_names)):
            task_loss = mse_loss(predictions[:, i], targets[:, i])
            
            # Apply uncertainty weighting
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * task_loss + self.log_vars[i]
            
            task_losses.append(weighted_loss)
        
        # Combine losses
        total_loss = torch.sum(torch.stack(task_losses))
        
        return total_loss

class MultitaskLearningPredictor:
    """Multitask learning predictor with hierarchical structure"""
    
    def __init__(self, input_dim=30, device='cpu'):
        self.input_dim = input_dim
        self.device = device
        
        # Initialize model
        self.model = HierarchicalMultitaskNet(input_dim=input_dim, shared_dim=128, dropout=0.2).to(device)
        
        # Custom loss function
        self.criterion = MultitaskLoss(learn_uncertainty=True)
        
        # Data preprocessing
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        
        # Target names
        self.target_names = [
            'kidney_left_delta_x', 'kidney_left_delta_y', 'kidney_left_delta_z',
            'kidney_right_delta_x', 'kidney_right_delta_y', 'kidney_right_delta_z'
        ]
    
    def prepare_data(self, df):
        """Prepare data for multitask learning"""
        print("Preparing data for multitask learning...")
        
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
        """Train the multitask model"""
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
                predictions = self.model(batch_features)
                loss = self.criterion(predictions, batch_targets)
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
                    
                    predictions = self.model(batch_features)
                    loss = self.criterion(predictions, batch_targets)
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
        """Train the multitask model"""
        print("Training multitask learning model...")
        
        # Create datasets and dataloaders
        train_dataset = MultitaskDataset(X_train, y_train)
        test_dataset = MultitaskDataset(X_test, y_test)
        
        # Split training data for validation
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Train model
        train_losses, val_losses = self.train_model(train_loader, val_loader, epochs)
        
        return train_losses, val_losses
    
    def predict(self, X, return_attention=False):
        """Make predictions using the multitask model"""
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            if return_attention:
                predictions, attention_weights = self.model(X_tensor, return_attention=True)
                return predictions.cpu().numpy(), attention_weights.cpu().numpy()
            else:
                predictions = self.model(X_tensor)
                return predictions.cpu().numpy()
    
    def evaluate(self, X_test, y_test):
        """Evaluate multitask model performance"""
        print("Evaluating multitask learning model...")
        
        predictions = self.predict(X_test)
        
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
        
        return results, overall_metrics
    
    def analyze_task_relationships(self, X_test, y_test):
        """Analyze relationships between different tasks"""
        print("\nAnalyzing task relationships...")
        
        predictions, attention_weights = self.predict(X_test, return_attention=True)
        
        # Calculate task correlations
        task_correlations = np.corrcoef(predictions.T)
        
        print("Task Correlation Matrix:")
        print("    " + "    ".join([f"{name[-4:]:>4}" for name in self.target_names]))
        
        for i, target1 in enumerate(self.target_names):
            row_str = f"{target1[-4:]:>4} "
            for j, target2 in enumerate(self.target_names):
                corr = task_correlations[i, j]
                row_str += f"{corr:>6.3f} "
            print(row_str)
        
        # Find strongest correlations
        correlations = []
        for i in range(len(self.target_names)):
            for j in range(i+1, len(self.target_names)):
                corr = task_correlations[i, j]
                correlations.append((self.target_names[i], self.target_names[j], abs(corr)))
        
        correlations.sort(key=lambda x: x[2], reverse=True)
        
        print("\nStrongest task correlations:")
        for target1, target2, corr in correlations[:3]:
            print(f"  {target1} <-> {target2}: {corr:.3f}")
        
        return task_correlations, attention_weights
    
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
    """Main training pipeline for multitask learning"""
    print("="*80)
    print("MULTITASK LEARNING PREDICTOR - PHASE 3")
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
        predictor = MultitaskLearningPredictor(input_dim=30, device=device)
        
        # Prepare data
        X_train, X_test, y_train, y_test = predictor.prepare_data(df)
        
        # Train model
        train_losses, val_losses = predictor.train(X_train, X_test, y_train, y_test, epochs=100, batch_size=16)
        
        # Evaluate model
        results, overall_metrics = predictor.evaluate(X_test, y_test)
        
        # Analyze task relationships
        task_correlations, attention_weights = predictor.analyze_task_relationships(X_test, y_test)
        
        # Generate report
        print("\n" + "="*80)
        print("MULTITASK LEARNING RESULTS")
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
        results_df.to_csv('multitask_learning_results.csv', index=False)
        
        # Save model
        predictor.save_model('multitask_learning_model.pth')
        
        print(f"\n✅ Multitask learning model training completed!")
        print(f"Results saved to multitask_learning_results.csv")
        print(f"Model saved to multitask_learning_model.pth")
        
    else:
        print("No valid datasets found!")

if __name__ == "__main__":
    main()

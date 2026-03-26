#!/usr/bin/env python3
"""
Compare All Ensemble Models for Kidney Displacement Prediction
Final comparison of all ensemble strategies
"""

import pandas as pd
import numpy as np
import glob
import os

def load_all_results():
    """Load all result files"""
    print("Loading all result files...")
    
    # Define result files to compare
    result_files = {
        'Best_Single': 'models_comparison_summary.csv',
        'Voting_Ensemble': 'ensemble_results.csv',
        'Adaptive_Ensemble': 'adaptive_ensemble_results.csv',
        'TargetSpecific_Ensemble': 'target_specific_ensemble_results.csv',
        'ErrorCorrection_Ensemble': 'error_correction_ensemble_results.csv'
    }
    
    all_data = []
    
    for strategy, filename in result_files.items():
        if os.path.exists(filename):
            print(f"Loading {filename}...")
            df = pd.read_csv(filename)
            
            # Add strategy column
            df['Strategy'] = strategy
            
            # Filter relevant columns
            if 'Model' in df.columns:
                # For ensemble results, filter to ensemble models only
                if strategy != 'Best_Single':
                    ensemble_models = df['Model'].str.contains('Ensemble', na=False)
                    df = df[ensemble_models]
                
                all_data.append(df)
        else:
            print(f"Warning: {filename} not found")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    else:
        return None

def calculate_best_performance(df):
    """Calculate best performance metrics"""
    print("\nCalculating best performance metrics...")
    
    # Group by strategy and calculate average metrics
    strategy_summary = df.groupby('Strategy').agg({
        'MAE': 'mean',
        'RMSE': 'mean',
        'R2': 'mean',
        'Error_5mm': 'mean',
        'Error_10mm': 'mean'
    }).round(3)
    
    # Sort by MAE (lower is better)
    strategy_summary = strategy_summary.sort_values('MAE')
    
    return strategy_summary

def find_best_model_per_target(df):
    """Find best model for each target"""
    print("\nFinding best model per target...")
    
    # Group by target and find best MAE
    best_per_target = df.loc[df.groupby('Target')['MAE'].idxmin()]
    
    return best_per_target

def create_comparison_table(strategy_summary):
    """Create comparison table"""
    print("\nCreating comparison table...")
    
    comparison_data = []
    
    for strategy, row in strategy_summary.iterrows():
        comparison_data.append({
            'Strategy': strategy,
            'Average MAE (mm)': row['MAE'],
            'Average RMSE (mm)': row['RMSE'],
            'Average R²': row['R2'],
            'Average <5mm Accuracy (%)': row['Error_5mm'],
            'Average <10mm Accuracy (%)': row['Error_10mm'],
            'Rank': strategy_summary.index.get_loc(strategy) + 1
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df

def analyze_improvements(df, best_single_mae):
    """Analyze improvements over best single model"""
    print("\nAnalyzing improvements...")
    
    improvements = {}
    
    for strategy in df['Strategy'].unique():
        if strategy == 'Best_Single':
            continue
            
        strategy_data = df[df['Strategy'] == strategy]
        avg_mae = strategy_data['MAE'].mean()
        
        improvement = ((best_single_mae - avg_mae) / best_single_mae) * 100
        improvements[strategy] = improvement
    
    return improvements

def generate_comprehensive_report(df, strategy_summary, best_per_target, improvements):
    """Generate comprehensive comparison report"""
    print("\n" + "="*80)
    print("COMPREHENSIVE ENSEMBLE COMPARISON REPORT")
    print("="*80)
    
    print(f"\nStrategy Performance Ranking (by MAE):")
    print("-" * 50)
    
    for i, (strategy, row) in enumerate(strategy_summary.iterrows(), 1):
        print(f"{i}. {strategy}:")
        print(f"   MAE: {row['MAE']:.3f} mm")
        print(f"   RMSE: {row['RMSE']:.3f} mm")
        print(f"   R²: {row['R2']:.3f}")
        print(f"   <5mm accuracy: {row['Error_5mm']:.1f}%")
        print(f"   <10mm accuracy: {row['Error_10mm']:.1f}%")
        print()
    
    print(f"\nBest Model Per Target:")
    print("-" * 50)
    
    for target in df['Target'].unique():
        target_data = best_per_target[best_per_target['Target'] == target]
        if len(target_data) > 0:
            best_row = target_data.iloc[0]
            print(f"{target}:")
            print(f"  Best Strategy: {best_row['Strategy']}")
            print(f"  Best Model: {best_row.get('Model', 'N/A')}")
            print(f"  MAE: {best_row['MAE']:.3f} mm")
            print(f"  R²: {best_row['R2']:.3f}")
            print()
    
    print(f"\nImprovement Analysis:")
    print("-" * 50)
    
    # Get best single model MAE for reference
    best_single_data = df[df['Strategy'] == 'Best_Single']
    best_single_mae = best_single_data['MAE'].mean()
    
    print(f"Reference (Best Single Model): {best_single_mae:.3f} mm")
    print()
    
    for strategy, improvement in improvements.items():
        status = "✅ IMPROVED" if improvement > 0 else "❌ WORSENED"
        print(f"{strategy}: {improvement:+.1f}% {status}")
    
    print(f"\nKey Findings:")
    print("-" * 50)
    
    # Best overall strategy
    best_strategy = strategy_summary.index[0]
    best_mae = strategy_summary.iloc[0]['MAE']
    
    print(f"1. Best Overall Strategy: {best_strategy}")
    print(f"   MAE: {best_mae:.3f} mm")
    print(f"   Improvement over best single: {((best_single_mae - best_mae) / best_single_mae) * 100:+.1f}%")
    print()
    
    # Most successful strategy
    successful_strategies = sum(1 for imp in improvements.values() if imp > 0)
    total_strategies = len(improvements)
    
    print(f"2. Success Rate: {successful_strategies}/{total_strategies} strategies improved")
    print(f"   Success rate: {(successful_strategies/total_strategies)*100:.1f}%")
    print()
    
    # Best performing targets
    target_performance = df.groupby('Target')['MAE'].min().sort_values()
    print(f"3. Easiest Targets to Predict:")
    for i, (target, mae) in enumerate(target_performance.head(3).items(), 1):
        print(f"   {i}. {target}: {mae:.3f} mm")
    print()
    
    print(f"4. Hardest Targets to Predict:")
    for i, (target, mae) in enumerate(target_performance.tail(3).items(), 1):
        print(f"   {i}. {target}: {mae:.3f} mm")

def save_final_comparison(comparison_df, best_per_target, filename='final_ensemble_comparison.csv'):
    """Save final comparison results"""
    print(f"\nSaving final comparison to {filename}...")
    
    # Combine comparison and best per target
    final_data = []
    
    # Add strategy comparison
    for _, row in comparison_df.iterrows():
        final_data.append({
            'Type': 'Strategy_Comparison',
            'Strategy': row['Strategy'],
            'Metric': 'Average_MAE',
            'Value': row['Average MAE (mm)'],
            'Rank': row['Rank']
        })
    
    # Add best per target
    for _, row in best_per_target.iterrows():
        final_data.append({
            'Type': 'Best_Per_Target',
            'Strategy': row['Strategy'],
            'Target': row['Target'],
            'Model': row.get('Model', 'N/A'),
            'MAE': row['MAE'],
            'R2': row['R2']
        })
    
    final_df = pd.DataFrame(final_data)
    final_df.to_csv(filename, index=False)
    print(f"Final comparison saved to {filename}")

def main():
    """Main comparison pipeline"""
    print("="*80)
    print("COMPREHENSIVE ENSEMBLE MODEL COMPARISON")
    print("="*80)
    
    # Load all results
    df = load_all_results()
    if df is None:
        print("No result files found!")
        return
    
    print(f"Loaded {len(df)} results from {df['Strategy'].nunique()} strategies")
    
    # Calculate strategy summary
    strategy_summary = calculate_best_performance(df)
    
    # Find best model per target
    best_per_target = find_best_model_per_target(df)
    
    # Analyze improvements
    best_single_mae = df[df['Strategy'] == 'Best_Single']['MAE'].mean()
    improvements = analyze_improvements(df, best_single_mae)
    
    # Create comparison table
    comparison_df = create_comparison_table(strategy_summary)
    
    # Generate comprehensive report
    generate_comprehensive_report(df, strategy_summary, best_per_target, improvements)
    
    # Save final comparison
    save_final_comparison(comparison_df, best_per_target)
    
    print(f"\n" + "="*80)
    print("ENSEMBLE COMPARISON COMPLETED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()

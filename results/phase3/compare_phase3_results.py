#!/usr/bin/env python3
"""
Compare Phase 3 Results - Advanced Research Approaches
Comprehensive comparison of all Phase 3 advanced models
"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path

def load_phase3_results():
    """Load all Phase 3 result files"""
    print("Loading Phase 3 result files...")
    
    # Define Phase 3 result files
    result_files = {
        'Neural_Network_Ensemble': 'neural_network_ensemble_results.csv',
        'Multitask_Learning': 'multitask_learning_results.csv',
        'Uncertainty_Quantification': 'uncertainty_quantification_results.csv',
        # Add other Phase 3 results as they become available
    }
    
    all_data = []
    
    for strategy, filename in result_files.items():
        if os.path.exists(filename):
            print(f"Loading {filename}...")
            df = pd.read_csv(filename)
            
            # Add strategy column
            df['Strategy'] = strategy
            
            # Filter relevant data
            if 'Target' in df.columns:
                all_data.append(df)
        else:
            print(f"Warning: {filename} not found")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    else:
        return None

def load_phase2_benchmark():
    """Load Phase 2 benchmark results"""
    print("Loading Phase 2 benchmark results...")
    
    # Load the best Phase 2 results
    benchmark_files = [
        'dynamic_adaptive_ensemble_results.csv',
        'phase2_enhancement_summary.csv'
    ]
    
    for filename in benchmark_files:
        if os.path.exists(filename):
            print(f"Loading benchmark: {filename}...")
            df = pd.read_csv(filename)
            
            if 'dynamic_adaptive_ensemble_results.csv' in filename:
                # Filter dynamic ensemble results
                dynamic_df = df[df['Model'] == 'Dynamic_Adaptive_Ensemble'].copy()
                return dynamic_df
            elif 'phase2_enhancement_summary.csv' in filename:
                # Get best strategy from comparison
                if 'Multivariate_Predictor' in df['Strategy'].values:
                    best_df = df[df['Strategy'] == 'Multivariate_Predictor'].copy()
                    return best_df
    
    return None

def calculate_improvements(phase3_df, benchmark_df):
    """Calculate improvements over Phase 2 benchmark"""
    print("Calculating improvements over Phase 2...")
    
    improvements = {}
    
    # Get benchmark MAE
    if benchmark_df is not None and 'MAE' in benchmark_df.columns:
        benchmark_mae = benchmark_df['MAE'].mean()
        print(f"Phase 2 Benchmark MAE: {benchmark_mae:.3f} mm")
        
        for strategy in phase3_df['Strategy'].unique():
            strategy_data = phase3_df[phase3_df['Strategy'] == strategy]
            
            if 'MAE' in strategy_data.columns:
                avg_mae = strategy_data['MAE'].mean()
                improvement = ((benchmark_mae - avg_mae) / benchmark_mae) * 100
                improvements[strategy] = improvement
                print(f"{strategy}: {avg_mae:.3f} mm ({improvement:+.1f}% vs Phase 2)")
    
    return improvements

def generate_phase3_report(phase3_df, benchmark_df, improvements):
    """Generate comprehensive Phase 3 report"""
    print("\n" + "="*80)
    print("PHASE 3 RESEARCH APPROACHES REPORT")
    print("="*80)
    
    print(f"\nPhase 3 Strategies Tested:")
    print("-" * 50)
    
    for strategy in phase3_df['Strategy'].unique():
        strategy_data = phase3_df[phase3_df['Strategy'] == strategy]
        
        if 'MAE' in strategy_data.columns:
            avg_mae = strategy_data['MAE'].mean()
            avg_r2 = strategy_data['R2'].mean() if 'R2' in strategy_data.columns else np.nan
            
            print(f"{strategy}:")
            print(f"  Average MAE: {avg_mae:.3f} mm")
            print(f"  Average R²: {avg_r2:.3f}")
            print(f"  Improvement: {improvements.get(strategy, 0):+.1f}%")
            print()
    
    print(f"\nPhase 3 vs Phase 2 Comparison:")
    print("-" * 50)
    
    if benchmark_df is not None:
        benchmark_mae = benchmark_df['MAE'].mean()
        print(f"Phase 2 Best (Dynamic Adaptive): {benchmark_mae:.3f} mm")
        
        best_phase3_mae = float('inf')
        best_strategy = None
        
        for strategy, improvement in improvements.items():
            strategy_data = phase3_df[phase3_df['Strategy'] == strategy]
            if 'MAE' in strategy_data.columns:
                avg_mae = strategy_data['MAE'].mean()
                if avg_mae < best_phase3_mae:
                    best_phase3_mae = avg_mae
                    best_strategy = strategy
        
        if best_strategy:
            total_improvement = ((benchmark_mae - best_phase3_mae) / benchmark_mae) * 100
            print(f"Phase 3 Best ({best_strategy}): {best_phase3_mae:.3f} mm")
            print(f"Total Improvement: {total_improvement:+.1f}%")
    
    print(f"\nKey Findings:")
    print("-" * 50)
    
    # Find best performing strategy
    best_strategy = None
    best_mae = float('inf')
    
    for strategy in phase3_df['Strategy'].unique():
        strategy_data = phase3_df[phase3_df['Strategy'] == strategy]
        if 'MAE' in strategy_data.columns:
            avg_mae = strategy_data['MAE'].mean()
            if avg_mae < best_mae:
                best_mae = avg_mae
                best_strategy = strategy
    
    if best_strategy:
        print(f"1. Best Phase 3 Strategy: {best_strategy}")
        print(f"   MAE: {best_mae:.3f} mm")
        print(f"   Improvement: {improvements.get(best_strategy, 0):+.1f}%")
    
    # Count successful strategies
    successful_strategies = sum(1 for imp in improvements.values() if imp > 0)
    total_strategies = len(improvements)
    
    print(f"\n2. Success Rate: {successful_strategies}/{total_strategies} strategies improved")
    print(f"   Success rate: {(successful_strategies/total_strategies)*100:.1f}%")
    
    # Analyze specific improvements
    print(f"\n3. Strategy Analysis:")
    
    for strategy, improvement in improvements.items():
        status = "✅ IMPROVED" if improvement > 0 else "❌ NO IMPROVEMENT"
        print(f"   {strategy}: {improvement:+.1f}% {status}")
    
    print(f"\n4. Research Insights:")
    
    if improvements:
        best_improvement = max(improvements.values())
        if best_improvement > 0:
            best_strategy = max(improvements, key=improvements.get)
            print(f"   → {best_strategy} shows most promise (+{best_improvement:.1f}% improvement)")
            print(f"   → Advanced deep learning approaches can enhance accuracy")
        else:
            print(f"   → Phase 2 Dynamic Adaptive remains the best approach")
            print(f"   → Research approaches need further refinement")
    else:
        print(f"   → Need more Phase 3 development work")
    
    print(f"\n5. Clinical Implications:")
    
    if improvements:
        any_improved = any(imp > 0 for imp in improvements.values())
        if any_improved:
            print(f"   → Advanced models show potential for clinical deployment")
            print(f"   → Uncertainty quantification provides confidence estimates")
            print(f"   → Multitask learning captures anatomical relationships")
        else:
            print(f"   → Phase 2 Dynamic Adaptive is recommended for clinical use")
            print(f"   → Research approaches require more development")
    
    print(f"\n6. Future Research Directions:")
    
    print(f"   → Hybrid models combining Phase 2 and Phase 3 approaches")
    print(f"   → Real-time learning from clinical feedback")
    print(f"   → Explainable AI for clinical decision support")
    print(f"   → Federated learning across institutions")

def create_phase3_summary_table(phase3_df, improvements):
    """Create summary table for Phase 3 results"""
    print(f"\nPhase 3 Summary Table:")
    print("-" * 80)
    
    summary_data = []
    
    for strategy in phase3_df['Strategy'].unique():
        strategy_data = phase3_df[phase3_df['Strategy'] == strategy]
        
        if 'MAE' in strategy_data.columns:
            avg_mae = strategy_data['MAE'].mean()
            avg_r2 = strategy_data['R2'].mean() if 'R2' in strategy_data.columns else np.nan
            improvement = improvements.get(strategy, 0)
            
            # Determine approach type
            approach_type = "Deep Learning" if "Neural" in strategy else "Advanced ML"
            if "Multitask" in strategy:
                approach_type = "Multitask Learning"
            elif "Uncertainty" in strategy:
                approach_type = "Bayesian/Probabilistic"
            
            summary_data.append({
                'Strategy': strategy,
                'Approach_Type': approach_type,
                'Average_MAE': avg_mae,
                'Average_R2': avg_r2,
                'Improvement_%': improvement,
                'Status': '✅ Improved' if improvement > 0 else '❌ No Change'
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Average_MAE')
    
    print(summary_df.to_string(index=False))
    
    return summary_df

def analyze_technical_complexity(phase3_df):
    """Analyze technical complexity of Phase 3 approaches"""
    print(f"\nTechnical Complexity Analysis:")
    print("-" * 50)
    
    complexity_analysis = {
        'Neural_Network_Ensemble': {
            'complexity': 'High',
            'training_time': 'Long',
            'interpretability': 'Low',
            'uncertainty': 'Medium',
            'scalability': 'Medium'
        },
        'Multitask_Learning': {
            'complexity': 'High',
            'training_time': 'Medium',
            'interpretability': 'Medium',
            'uncertainty': 'Low',
            'scalability': 'High'
        },
        'Uncertainty_Quantification': {
            'complexity': 'Very High',
            'training_time': 'Very Long',
            'interpretability': 'High',
            'uncertainty': 'Very High',
            'scalability': 'Low'
        }
    }
    
    for strategy in phase3_df['Strategy'].unique():
        if strategy in complexity_analysis:
            analysis = complexity_analysis[strategy]
            print(f"{strategy}:")
            for aspect, level in analysis.items():
                print(f"  {aspect}: {level}")
            print()

def save_phase3_results(summary_df, filename='phase3_research_summary.csv'):
    """Save Phase 3 results to CSV"""
    print(f"\nSaving Phase 3 summary to {filename}...")
    summary_df.to_csv(filename, index=False)
    print(f"Phase 3 summary saved to {filename}")

def generate_research_recommendations(improvements):
    """Generate research recommendations based on results"""
    print(f"\n" + "="*80)
    print("RESEARCH RECOMMENDATIONS")
    print("="*80)
    
    print(f"\nBased on Phase 3 results:")
    
    if any(imp > 0 for imp in improvements.values()):
        print(f"✅ Advanced approaches show promise for further development")
        
        # Find most promising approach
        best_strategy = max(improvements, key=improvements.get) if improvements else None
        best_improvement = improvements.get(best_strategy, 0) if best_strategy else 0
        
        print(f"\n🎯 Priority Research Directions:")
        print(f"1. {best_strategy} (best improvement: +{best_improvement:.1f}%)")
        print(f"   → Refine hyperparameters and architecture")
        print(f"   → Combine with Phase 2 enhanced features")
        print(f"   → Clinical validation with real-world data")
        
        print(f"\n2. Hybrid Approaches:")
        print(f"   → Combine Phase 2 Dynamic Adaptive with Phase 3 neural networks")
        print(f"   → Ensemble of traditional and deep learning models")
        print(f"   → Multi-stage prediction with uncertainty estimation")
        
        print(f"\n3. Clinical Integration:")
        print(f"   → Real-time inference optimization")
        print(f"   → Explainable AI for clinical trust")
        print(f"   → Continuous learning from clinical feedback")
        
    else:
        print(f"❌ Advanced approaches need more development")
        
        print(f"\n🎯 Recommended Research Path:")
        print(f"1. Stick with Phase 2 Dynamic Adaptive for production")
        print(f"2. Refine Phase 3 approaches with more data")
        print(f"3. Investigate hybrid approaches")
        print(f"4. Focus on interpretability and clinical acceptance")
    
    print(f"\n📊 Next Steps:")
    print(f"→ Collect more diverse training data")
    print(f"→ Implement real-time learning pipelines")
    print(f"→ Develop clinical validation protocols")
    print(f"→ Create explainable AI interfaces")
    print(f"→ Plan multi-center clinical trials")

def main():
    """Main Phase 3 comparison pipeline"""
    print("="*80)
    print("PHASE 3 RESEARCH APPROACHES COMPARISON")
    print("="*80)
    
    # Load Phase 3 results
    phase3_df = load_phase3_results()
    if phase3_df is None:
        print("No Phase 3 results found!")
        return
    
    print(f"Loaded {len(phase3_df)} Phase 3 results")
    
    # Load Phase 2 benchmark
    benchmark_df = load_phase2_benchmark()
    
    # Calculate improvements
    improvements = calculate_improvements(phase3_df, benchmark_df)
    
    # Generate report
    generate_phase3_report(phase3_df, benchmark_df, improvements)
    
    # Create summary table
    summary_df = create_phase3_summary_table(phase3_df, improvements)
    
    # Analyze technical complexity
    analyze_technical_complexity(phase3_df)
    
    # Generate research recommendations
    generate_research_recommendations(improvements)
    
    # Save results
    save_phase3_results(summary_df)
    
    print(f"\n" + "="*80)
    print("PHASE 3 COMPARISON COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()

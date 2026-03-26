#!/usr/bin/env python3
"""
Phase 3 Research Pipeline Runner
Execute all Phase 3 research approaches and generate comprehensive comparison
"""

import subprocess
import sys
import os
import time
from pathlib import Path
import pandas as pd
import numpy as np

def run_research_script(script_name, description):
    """Run a research script and capture results"""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=3600)
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            print(f"⏱️  Execution time: {execution_time:.2f} seconds")
            
            # Show key output
            output_lines = result.stdout.split('\n')
            key_lines = [line for line in output_lines 
                        if 'MAE:' in line or 'Average MAE:' in line or 'Results saved to' in line]
            
            if key_lines:
                print(f"📊 Key results:")
                for line in key_lines[-5:]:  # Show last 5 key lines
                    print(f"   {line}")
            
            return True, execution_time, result.stdout
        else:
            print(f"❌ {description} failed!")
            print(f"Error: {result.stderr}")
            return False, execution_time, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out after 3600 seconds")
        return False, 3600, "Timeout"
    except Exception as e:
        print(f"💥 {description} crashed: {str(e)}")
        return False, 0, str(e)

def check_dependencies():
    """Check if required dependencies are available"""
    print("Checking Phase 3 dependencies...")
    
    required_packages = [
        'torch', 'pandas', 'numpy', 'sklearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            print(f"❌ {package} missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {missing_packages}")
        print("Install with: pip install -r requirements_phase3.txt")
        return False
    
    print("✅ All dependencies available!")
    return True

def check_data_availability():
    """Check if required data files are available"""
    print("Checking data availability...")
    
    data_files = [
        'data/vybor_unified_features.csv',
        'data/kits19_medical_grade_features.csv'
    ]
    
    missing_files = []
    
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} available")
        else:
            print(f"❌ {file_path} missing")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing data files: {missing_files}")
        return False
    
    print("✅ All data files available!")
    return True

def create_results_directory():
    """Create results directory for Phase 3"""
    results_dir = Path("results/phase3")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Results directory: {results_dir}")
    return results_dir

def run_phase3_research():
    """Execute complete Phase 3 research pipeline"""
    print("="*80)
    print("PHASE 3 RESEARCH PIPELINE")
    print("="*80)
    print("Advanced Deep Learning Approaches for Kidney Displacement Prediction")
    print("="*80)
    
    # Check prerequisites
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install required packages.")
        return False
    
    if not check_data_availability():
        print("\n❌ Data availability check failed. Please ensure data files are present.")
        return False
    
    # Create results directory
    results_dir = create_results_directory()
    
    # Define research scripts
    research_scripts = [
        {
            'script': 'neural_network_ensemble.py',
            'description': 'Neural Network Ensemble with Attention Mechanisms',
            'priority': 1
        },
        {
            'script': 'multitask_learning_predictor.py',
            'description': 'Multitask Learning with Hierarchical Structure',
            'priority': 2
        },
        {
            'script': 'uncertainty_quantification_predictor.py',
            'description': 'Uncertainty Quantification with Bayesian Methods',
            'priority': 3
        }
    ]
    
    # Execute research scripts
    results = []
    total_start_time = time.time()
    
    for script_info in research_scripts:
        script_name = script_info['script']
        description = script_info['description']
        priority = script_info['priority']
        
        print(f"\n🔬 Starting Priority {priority} research...")
        
        success, execution_time, output = run_research_script(script_name, description)
        
        results.append({
            'script': script_name,
            'description': description,
            'priority': priority,
            'success': success,
            'execution_time': execution_time,
            'output': output
        })
        
        if not success:
            print(f"⚠️  {description} failed, continuing with next script...")
    
    total_execution_time = time.time() - total_start_time
    
    # Generate summary report
    generate_phase3_summary(results, total_execution_time, results_dir)
    
    # Run comparison if we have successful results
    successful_results = [r for r in results if r['success']]
    if len(successful_results) > 1:
        print(f"\n📊 Running Phase 3 comparison...")
        run_comparison_script(successful_results, results_dir)
    else:
        print(f"\n⚠️  Not enough successful results for comparison")
    
    print(f"\n{'='*80}")
    print("PHASE 3 RESEARCH PIPELINE COMPLETED")
    print(f"{'='*80}")
    print(f"Total execution time: {total_execution_time:.2f} seconds")
    print(f"Successful scripts: {len(successful_results)}/{len(results)}")
    print(f"Results saved to: {results_dir}")
    
    return True

def generate_phase3_summary(results, total_time, results_dir):
    """Generate summary of Phase 3 research results"""
    print(f"\n📝 Generating Phase 3 summary...")
    
    summary_lines = [
        "PHASE 3 RESEARCH SUMMARY",
        "=" * 50,
        f"Total execution time: {total_time:.2f} seconds",
        f"Scripts executed: {len(results)}",
        f"Successful: {len([r for r in results if r['success']])}",
        f"Failed: {len([r for r in results if not r['success']])}",
        "",
        "SCRIPT RESULTS:",
        "-" * 30
    ]
    
    for result in results:
        status = "SUCCESS" if result['success'] else "FAILED"
        summary_lines.append(f"{result['description']}: {status}")
        summary_lines.append(f"  Execution time: {result['execution_time']:.2f}s")
        
        # Extract MAE if available
        if result['success'] and 'Average MAE:' in result['output']:
            lines = result['output'].split('\n')
            for line in lines:
                if 'Average MAE:' in line:
                    summary_lines.append(f"  {line.strip()}")
                    break
        
        summary_lines.append("")
    
    # Save summary
    summary_file = results_dir / "phase3_research_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"📄 Summary saved to: {summary_file}")
    
    # Print summary
    print("\n" + '\n'.join(summary_lines[:20]))  # Show first 20 lines

def run_comparison_script(successful_results, results_dir):
    """Run Phase 3 comparison script"""
    try:
        print(f"📊 Comparing Phase 3 results...")
        
        # Move result files to results directory
        result_files = [
            'neural_network_ensemble_results.csv',
            'multitask_learning_results.csv',
            'uncertainty_quantification_results.csv'
        ]
        
        for file_name in result_files:
            if os.path.exists(file_name):
                dest = results_dir / file_name
                os.rename(file_name, dest)
                print(f"📁 Moved {file_name} to {dest}")
        
        # Run comparison
        success, _, output = run_research_script('compare_phase3_results.py', 'Phase 3 Results Comparison')
        
        if success:
            # Move comparison results
            comparison_files = [
                'phase3_research_summary.csv'
            ]
            
            for file_name in comparison_files:
                if os.path.exists(file_name):
                    dest = results_dir / file_name
                    os.rename(file_name, dest)
                    print(f"📁 Moved {file_name} to {dest}")
            
            print(f"✅ Phase 3 comparison completed!")
        else:
            print(f"⚠️  Phase 3 comparison failed")
            
    except Exception as e:
        print(f"💥 Comparison failed: {str(e)}")

def print_research_recommendations(results):
    """Print research recommendations based on results"""
    print(f"\n🎯 RESEARCH RECOMMENDATIONS:")
    print("=" * 50)
    
    successful_results = [r for r in results if r['success']]
    
    if len(successful_results) == 0:
        print("❌ No successful research approaches")
        print("→ Debug data and dependency issues")
        print("→ Consider simpler approaches first")
        return
    
    if len(successful_results) == 1:
        print("📊 Only one approach succeeded")
        print("→ Focus on refining the successful approach")
        print("→ Investigate why others failed")
        return
    
    # Find best performing approach
    best_mae = float('inf')
    best_approach = None
    
    for result in successful_results:
        if 'Average MAE:' in result['output']:
            lines = result['output'].split('\n')
            for line in lines:
                if 'Average MAE:' in line:
                    try:
                        mae_str = line.split('Average MAE:')[1].strip().split()[0]
                        mae = float(mae_str)
                        if mae < best_mae:
                            best_mae = mae
                            best_approach = result['description']
                    except:
                        pass
    
    if best_approach:
        print(f"🏆 Best performing approach: {best_approach}")
        print(f"📈 Best MAE: {best_mae:.3f} mm")
        print(f"\n🎯 Next steps:")
        print(f"→ Focus research on {best_approach}")
        print(f"→ Combine with Phase 2 enhanced features")
        print(f"→ Investigate hyperparameter optimization")
        print(f"→ Plan clinical validation studies")
    
    print(f"\n🔬 Future research directions:")
    print(f"→ Hybrid models combining successful approaches")
    print(f"→ Real-time learning from clinical feedback")
    print(f"→ Explainable AI for clinical acceptance")
    print(f"→ Multi-center validation studies")

def main():
    """Main Phase 3 research pipeline"""
    try:
        # Run complete Phase 3 research
        success = run_phase3_research()
        
        if success:
            print(f"\n🎉 Phase 3 research completed successfully!")
            print(f"📊 Check results directory for detailed outputs")
            print(f"🔬 Review comparison results for insights")
        else:
            print(f"\n⚠️  Phase 3 research completed with issues")
            print(f"🔧 Check error messages and fix dependencies")
        
        print(f"\n📚 For detailed research methodology, see: README_PHASE3_RESEARCH.md")
        
    except KeyboardInterrupt:
        print(f"\n🛑 Phase 3 research interrupted by user")
    except Exception as e:
        print(f"\n💥 Phase 3 research pipeline failed: {str(e)}")

if __name__ == "__main__":
    main()

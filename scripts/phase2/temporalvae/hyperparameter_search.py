#!/usr/bin/env python3
"""
Hyperparameter Grid Search for Temporal VAE
Tests different combinations of beta and KL warmup

"""

import subprocess
import json
from pathlib import Path
from datetime import datetime

# Test configurations
TEST_CONFIGS = [
    # Config 1: Very low beta, standard warmup
    {
        'name': 'low_beta_v1',
        'beta': 0.05,
        'kl_warmup_epochs': 50,
        'recon_weight': 3.0,
        'description': 'Very low beta (0.05) to strongly prevent posterior collapse'
    },
    
    # Config 2: Very low beta, long warmup
    {
        'name': 'low_beta_long_warmup',
        'beta': 0.05,
        'kl_warmup_epochs': 100,
        'recon_weight': 3.0,
        'description': 'Very low beta + extended warmup (100 epochs)'
    },
    
    # Config 3: Extremely low beta
    {
        'name': 'extreme_low_beta',
        'beta': 0.01,
        'kl_warmup_epochs': 50,
        'recon_weight': 3.0,
        'description': 'Extremely low beta (0.01) - maximum variance preservation'
    },
    
    # Config 4: Higher recon weight
    {
        'name': 'high_recon_weight',
        'beta': 0.1,
        'kl_warmup_epochs': 50,
        'recon_weight': 5.0,
        'description': 'Standard beta but higher reconstruction weight'
    },
    
    # Config 5: Balanced approach
    {
        'name': 'balanced',
        'beta': 0.1,
        'kl_warmup_epochs': 75,
        'recon_weight': 4.0,
        'description': 'Balanced parameters - moderate on all fronts'
    }
]

def modify_config_in_script(script_path, config):
    """Modify CONFIG dictionary in the training script"""
    with open(script_path, 'r') as f:
        lines = f.readlines()
    
    modified_lines = []
    in_config = False
    
    for line in lines:
        if 'CONFIG = {' in line:
            in_config = True
            modified_lines.append(line)
        elif in_config and '}' in line and 'CONFIG' in ''.join(modified_lines[-10:]):
            # End of CONFIG dict
            in_config = False
            modified_lines.append(line)
        elif in_config:
            # Modify relevant parameters
            if "'beta':" in line:
                modified_lines.append(f"    'beta': {config['beta']},\n")
            elif "'kl_warmup_epochs':" in line:
                modified_lines.append(f"    'kl_warmup_epochs': {config['kl_warmup_epochs']},\n")
            elif "'recon_weight':" in line:
                modified_lines.append(f"    'recon_weight': {config['recon_weight']},\n")
            elif "'model_name':" in line:
                modified_lines.append(f"    'model_name': 'temporal_vae_{config['name']}',\n")
            elif "'output_dir':" in line:
                output_dir = f"Path.home() / 'generative-ai-workload-modeling/outputs/temporal_vae_{config['name']}'"
                modified_lines.append(f"    'output_dir': {output_dir},\n")
            else:
                modified_lines.append(line)
        else:
            modified_lines.append(line)
    
    return ''.join(modified_lines)

def run_experiment(config, script_path, temp_script_path):
    """Run a single experiment with given config"""
    print("\n" + "="*100)
    print(f"TESTING: {config['name']}")
    print("="*100)
    print(f"Description: {config['description']}")
    print(f"Parameters:")
    print(f"  beta: {config['beta']}")
    print(f"  kl_warmup_epochs: {config['kl_warmup_epochs']}")
    print(f"  recon_weight: {config['recon_weight']}")
    print("-"*100)
    
    # Modify script with config
    modified_script = modify_config_in_script(script_path, config)
    
    # Write temporary script
    with open(temp_script_path, 'w') as f:
        f.write(modified_script)
    
    # Run experiment
    try:
        result = subprocess.run(
            ['python', str(temp_script_path)],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            print(f"\nExperiment {config['name']} completed successfully")
            return True
        else:
            print(f"\nExperiment {config['name']} failed with error:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\nExperiment {config['name']} timed out after 1 hour")
        return False
    except Exception as e:
        print(f"\nExperiment {config['name']} failed with exception: {e}")
        return False

def summarize_results(output_base_dir):
    """Create summary of all experiment results"""
    print("\n" + "="*100)
    print("EXPERIMENT SUMMARY")
    print("="*100)
    
    results_summary = []
    
    for config in TEST_CONFIGS:
        output_dir = Path(output_base_dir) / f"temporal_vae_{config['name']}"
        results_file = output_dir / 'evaluation_results.json'
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            summary = {
                'name': config['name'],
                'beta': config['beta'],
                'kl_warmup': config['kl_warmup_epochs'],
                'recon_weight': config['recon_weight'],
                'variance_ratio': results.get('overall', {}).get('variance_ratio', 0),
                'avg_dtw': results.get('temporal', {}).get('avg_dtw', 0),
                'avg_acf_error': results.get('temporal', {}).get('avg_acf_error', 0)
            }
            
            results_summary.append(summary)
    
    if results_summary:
        # Print table
        print(f"\n{'Config':<25} {'Beta':<10} {'KL Warmup':<12} {'Recon W':<10} {'Var Ratio':<12} {'Avg DTW':<12} {'ACF Error':<12}")
        print("-"*100)
        
        for r in results_summary:
            print(f"{r['name']:<25} {r['beta']:<10.2f} {r['kl_warmup']:<12d} {r['recon_weight']:<10.1f} "
                  f"{r['variance_ratio']:<12.4f} {r['avg_dtw']:<12.4f} {r['avg_acf_error']:<12.6f}")
        
        # Find best config
        best_config = max(results_summary, key=lambda x: x['variance_ratio'])
        
        print("\n" + "="*100)
        print("BEST CONFIGURATION")
        print("="*100)
        print(f"Name: {best_config['name']}")
        print(f"Beta: {best_config['beta']}")
        print(f"KL Warmup: {best_config['kl_warmup']}")
        print(f"Recon Weight: {best_config['recon_weight']}")
        print(f"\nResults:")
        print(f"  Variance Ratio: {best_config['variance_ratio']:.4f}")
        print(f"  Avg DTW: {best_config['avg_dtw']:.4f}")
        print(f"  ACF Error: {best_config['avg_acf_error']:.6f}")
        
        # Save summary
        summary_file = Path(output_base_dir) / 'hyperparameter_search_summary.json'
        with open(summary_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'configs_tested': TEST_CONFIGS,
                'results': results_summary,
                'best_config': best_config
            }, f, indent=2)
        
        print(f"\nSummary saved to: {summary_file}")
    
    print("\n" + "="*100)

def main():
    """Run hyperparameter grid search"""
    print("="*100)
    print("TEMPORAL VAE - HYPERPARAMETER GRID SEARCH")
    print("="*100)
    print(f"\nTesting {len(TEST_CONFIGS)} configurations")
    print("This will take several hours (approximately 30-60 minutes per config)")
    print("\nConfigurations to test:")
    
    for i, config in enumerate(TEST_CONFIGS, 1):
        print(f"{i}. {config['name']}: {config['description']}")
    
    # Paths
    script_path = Path(__file__).parent / 'temporal_vae_improved.py'
    temp_script_path = Path(__file__).parent / 'temp_training_script.py'
    output_base_dir = Path.home() / 'generative-ai-workload-modeling/outputs'
    
    if not script_path.exists():
        print(f"\nError: Training script not found at {script_path}")
        return
    
    # Run experiments
    print("\nStarting experiments...")
    
    successful = 0
    failed = 0
    
    for config in TEST_CONFIGS:
        success = run_experiment(config, script_path, temp_script_path)
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Clean up temp script
    if temp_script_path.exists():
        temp_script_path.unlink()
    
    # Print results
    print("\n" + "="*100)
    print("GRID SEARCH COMPLETE")
    print("="*100)
    print(f"\nSuccessful experiments: {successful}/{len(TEST_CONFIGS)}")
    print(f"Failed experiments: {failed}/{len(TEST_CONFIGS)}")
    
    # Summarize results
    if successful > 0:
        summarize_results(output_base_dir)
    
    print("\nTo compare all models (including previous TimeVAE versions), run:")
    print("  python compare_vae_models.py")

if __name__ == '__main__':
    main()
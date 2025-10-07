#!/usr/bin/env python3
"""Run a comprehensive parameter study to explore effects on metrics."""
import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import datetime
from itertools import product

def run_experiment(params):
    """Run a single experiment with given parameters."""
    cmd = [
        "modal", "run", "src/modal/sd3_modal.py",
        "--prompt-1", params["prompt_1"],
        "--prompt-2", params["prompt_2"],
        "--switch-step", str(params["switch_step"]),
        "--seed", str(params["seed"]),
        "--scheduler", params["scheduler"],
        "--num-inference-steps", str(params.get("num_steps", 50)),
        "--guidance-scale", str(params.get("guidance_scale", 7.0)),
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            # Parse metrics from output
            metrics = {"status": "success"}
            output_lines = result.stdout.split('\n')
            
            for line in output_lines:
                if "High freq change:" in line:
                    metrics["hf_change"] = float(line.split(":")[1].strip())
                elif "CLIP delta:" in line:
                    metrics["clip_delta"] = float(line.split(":")[1].strip())
                elif "LPIPS to reference 1:" in line:
                    metrics["lpips_ref1"] = float(line.split(":")[1].strip())
                elif "LPIPS to reference 2:" in line:
                    metrics["lpips_ref2"] = float(line.split(":")[1].strip())
                elif "Attention variance (mean):" in line:
                    metrics["attention_variance"] = float(line.split(":")[1].strip())
                elif "Perceptual blend ratio:" in line:
                    metrics["blend_ratio"] = float(line.split(":")[1].strip().rstrip('%'))
            
            return {**params, **metrics}
        else:
            return {**params, "status": "failed", "error": result.stderr[:200]}
            
    except Exception as e:
        return {**params, "status": "error", "error": str(e)}

def main():
    print("🔬 Starting Parameter Study")
    print("=" * 50)
    
    # Define parameter space
    experiments = []
    
    # 1. Switch Step Study (most important)
    print("📋 Experiment 1: Switch Step Effects")
    base_params = {
        "prompt_1": "A peaceful mountain landscape",
        "prompt_2": "A busy city street at night", 
        "seed": 42,
        "scheduler": "FlowMatchEulerDiscreteScheduler"
    }
    
    for switch_step in [5, 10, 15, 20, 25, 30, 35, 40, 45]:
        experiments.append({**base_params, "switch_step": switch_step, "study": "switch_step"})
    
    # 2. Scheduler Comparison
    print("📋 Experiment 2: Scheduler Effects")
    schedulers = [
        "FlowMatchEulerDiscreteScheduler",
        "DPMSolverMultistepScheduler", 
        "EulerDiscreteScheduler",
        "DDIMScheduler",
        "HeunDiscreteScheduler"
    ]
    
    for scheduler in schedulers:
        experiments.append({
            **base_params, 
            "switch_step": 15,  # Early switching
            "scheduler": scheduler,
            "study": "scheduler"
        })
    
    # 3. Prompt Similarity Study
    print("📋 Experiment 3: Prompt Similarity Effects")
    prompt_pairs = [
        # Very similar
        ("A red car", "A blue car"),
        # Somewhat similar  
        ("A mountain landscape", "A forest landscape"),
        # Very different
        ("A peaceful garden", "A space station"),
        # Abstract vs concrete
        ("Love and happiness", "A concrete building")
    ]
    
    for i, (p1, p2) in enumerate(prompt_pairs):
        experiments.append({
            "prompt_1": p1,
            "prompt_2": p2,
            "switch_step": 20,
            "seed": 42,
            "scheduler": "FlowMatchEulerDiscreteScheduler",
            "study": "prompt_similarity",
            "similarity_level": ["very_similar", "somewhat_similar", "very_different", "abstract_concrete"][i]
        })
    
    # 4. Guidance Scale Study
    print("📋 Experiment 4: Guidance Scale Effects")
    for guidance in [3.0, 5.0, 7.0, 9.0, 12.0]:
        experiments.append({
            **base_params,
            "switch_step": 20,
            "guidance_scale": guidance,
            "study": "guidance_scale"
        })
    
    print(f"\n🚀 Running {len(experiments)} experiments...")
    
    # Run experiments
    results = []
    for i, params in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] {params['study']}: ", end="", flush=True)
        
        if params['study'] == 'switch_step':
            print(f"step {params['switch_step']}")
        elif params['study'] == 'scheduler':
            print(f"{params['scheduler']}")
        elif params['study'] == 'prompt_similarity':
            print(f"{params['similarity_level']}")
        elif params['study'] == 'guidance_scale':
            print(f"guidance {params.get('guidance_scale', 7.0)}")
        
        result = run_experiment(params)
        results.append(result)
        
        if result.get("status") == "success":
            print(f"  ✓ CLIP Δ: {result.get('clip_delta', 0):+.3f}, Blend: {result.get('blend_ratio', 0):.1f}%")
        else:
            print(f"  ✗ Failed: {result.get('error', 'Unknown')[:50]}")
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"parameter_study_results_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump({
            "study": "comprehensive_parameter_analysis",
            "timestamp": timestamp,
            "total_experiments": len(experiments),
            "results": results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Create DataFrame for analysis
    df = pd.DataFrame([r for r in results if r.get("status") == "success"])
    
    if len(df) > 0:
        print(f"\n📊 Successfully completed: {len(df)}/{len(results)} experiments")
        print(f"📈 Ready for visualization! Use: python visualize_parameter_study.py {results_file}")
        
        # Quick preview
        print("\n🔍 Quick Preview:")
        for study in df['study'].unique():
            study_df = df[df['study'] == study]
            print(f"  {study}: {len(study_df)} experiments")
            if 'clip_delta' in study_df.columns:
                print(f"    CLIP delta range: {study_df['clip_delta'].min():.3f} to {study_df['clip_delta'].max():.3f}")
    else:
        print("❌ No successful experiments to analyze")
    
    return results_file

if __name__ == "__main__":
    results_file = main()
#!/usr/bin/env python3
"""Run a mini parameter study for quick visualization testing."""
import subprocess
import json
import datetime

def run_experiment(params):
    """Run a single experiment with given parameters."""
    cmd = [
        "modal", "run", "src/modal/sd3_modal.py",
        "--prompt-1", params["prompt_1"],
        "--prompt-2", params["prompt_2"],
        "--switch-step", str(params["switch_step"]),
        "--seed", str(params["seed"]),
        "--scheduler", params["scheduler"],
    ]
    
    try:
        print(f"  Running: {params['scheduler']} at step {params['switch_step']}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
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
    print("🔬 Running Mini Parameter Study for Visualization")
    print("=" * 50)
    
    # Define a small set of experiments
    experiments = []
    
    # 1. Switch step study (5 experiments)
    base_params = {
        "prompt_1": "A mountain landscape",
        "prompt_2": "A city street", 
        "seed": 42,
        "scheduler": "FlowMatchEulerDiscreteScheduler"
    }
    
    for switch_step in [5, 15, 25, 35, 45]:
        experiments.append({**base_params, "switch_step": switch_step, "study": "switch_step"})
    
    # 2. Scheduler comparison (3 experiments)
    schedulers = [
        "FlowMatchEulerDiscreteScheduler",
        "DPMSolverMultistepScheduler", 
        "EulerDiscreteScheduler",
    ]
    
    for scheduler in schedulers:
        experiments.append({
            **base_params, 
            "switch_step": 20,  # Mid switching
            "scheduler": scheduler,
            "study": "scheduler"
        })
    
    print(f"🚀 Running {len(experiments)} experiments...")
    
    # Run experiments
    results = []
    for i, params in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] {params['study']}: ", end="")
        
        result = run_experiment(params)
        results.append(result)
        
        if result.get("status") == "success":
            print(f"✓ CLIP: {result.get('clip_delta', 0):+.3f}, HF: {result.get('hf_change', 0):+.3f}")
        else:
            print(f"✗ Failed")
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"mini_study_results_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump({
            "study": "mini_parameter_analysis",
            "timestamp": timestamp,
            "total_experiments": len(experiments),
            "results": results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Run visualization
    successful = [r for r in results if r.get("status") == "success"]
    if len(successful) >= 3:
        print(f"\n📊 {len(successful)} successful experiments - running visualization...")
        
        viz_cmd = ["python", "visualize_parameter_study.py", results_file]
        subprocess.run(viz_cmd)
    else:
        print(f"\n❌ Only {len(successful)} successful experiments - need at least 3 for visualization")
    
    return results_file

if __name__ == "__main__":
    main()
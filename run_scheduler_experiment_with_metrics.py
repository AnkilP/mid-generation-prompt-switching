#!/usr/bin/env python3
"""Run scheduler experiment with comprehensive metrics analysis."""
import subprocess
import sys
import json
import datetime
from pathlib import Path

# List of all schedulers to test
schedulers = [
    "FlowMatchEulerDiscreteScheduler",
    "FlowMatchHeunDiscreteScheduler", 
    "DEISMultistepScheduler",
    "DPMSolverMultistepScheduler",
    "DPMSolverSinglestepScheduler",
    "KDPM2DiscreteScheduler",
    "KDPM2AncestralDiscreteScheduler",
    "EulerDiscreteScheduler",
    "EulerAncestralDiscreteScheduler",
    "HeunDiscreteScheduler",
    "PNDMScheduler",
    "DDIMScheduler",
    "DDPMScheduler",
    "LCMScheduler",
]

# Experiment parameters
prompt_1 = "A serene mountain landscape with snow peaks"
prompt_2 = "A bustling city street at night with neon lights"
switch_step = 10
seed = 42

print(f"\n{'='*60}")
print(f"Running scheduler experiment with comprehensive metrics")
print(f"Prompt 1: {prompt_1}")
print(f"Prompt 2: {prompt_2}")
print(f"Switch step: {switch_step}")
print(f"Total schedulers to test: {len(schedulers)}")
print(f"{'='*60}\n")

results = []

# Run experiment for each scheduler
for i, scheduler in enumerate(schedulers, 1):
    print(f"\n[{i}/{len(schedulers)}] Testing scheduler: {scheduler}")
    print("-" * 40)
    
    try:
        # Build the modal run command
        cmd = [
            "modal", "run", "src/modal/sd3_modal.py",
            "--prompt-1", prompt_1,
            "--prompt-2", prompt_2,
            "--switch-step", str(switch_step),
            "--seed", str(seed),
            "--scheduler", scheduler,
        ]
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Success with {scheduler}")
            
            # Parse output to extract metrics
            output_lines = result.stdout.split('\n')
            
            # Initialize metrics
            metrics = {
                "scheduler": scheduler,
                "status": "success",
                "hf_change": None,
                "clip_delta": None,
                "lpips_ref1": None,
                "lpips_ref2": None,
                "attention_variance": None,
                "blend_ratio": None,
            }
            
            # Parse metrics from output
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
            
            results.append(metrics)
            
            # Print summary
            if metrics["hf_change"] is not None:
                print(f"  HF change: {metrics['hf_change']:+.4f}")
            if metrics["clip_delta"] is not None:
                print(f"  CLIP delta: {metrics['clip_delta']:+.3f}")
            if metrics["lpips_ref1"] is not None and metrics["lpips_ref2"] is not None:
                print(f"  LPIPS: R1={metrics['lpips_ref1']:.3f}, R2={metrics['lpips_ref2']:.3f}")
            if metrics["blend_ratio"] is not None:
                print(f"  Blend ratio: {metrics['blend_ratio']:.1f}%")
                
        else:
            print(f"❌ Failed with {scheduler}")
            print(f"Error: {result.stderr[:200]}...")
            results.append({
                "scheduler": scheduler,
                "status": "failed",
                "error": result.stderr
            })
            
    except Exception as e:
        print(f"❌ Exception with {scheduler}: {str(e)}")
        results.append({
            "scheduler": scheduler,
            "status": "error",
            "error": str(e)
        })

# Save results
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f"scheduler_experiment_results_{timestamp}.json"

with open(results_file, "w") as f:
    json.dump({
        "experiment": "scheduler_comparison_with_metrics",
        "switch_step": switch_step,
        "prompt_1": prompt_1,
        "prompt_2": prompt_2,
        "seed": seed,
        "timestamp": timestamp,
        "results": results,
    }, f, indent=2)

print(f"\nResults saved to: {results_file}")

# Summary analysis
print(f"\n{'='*60}")
print("EXPERIMENT SUMMARY")
print(f"{'='*60}")

successful = [r for r in results if r.get("status") == "success" and r.get("clip_delta") is not None]
print(f"Total schedulers tested: {len(schedulers)}")
print(f"Successful runs with metrics: {len(successful)}")
print(f"Failed runs: {len(results) - len(successful)}")

if successful:
    # Sort by different metrics
    print("\n1. BEST BY CLIP ALIGNMENT (closest to intended prompt):")
    sorted_by_clip = sorted(successful, key=lambda x: abs(x.get('clip_delta', 0)))
    for i, r in enumerate(sorted_by_clip[:3], 1):
        print(f"   {i}. {r['scheduler']}: Δ={r.get('clip_delta', 0):+.3f}")
    
    print("\n2. BEST BY PERCEPTUAL QUALITY (lowest LPIPS):")
    sorted_by_lpips = sorted(successful, key=lambda x: min(x.get('lpips_ref1', 1), x.get('lpips_ref2', 1)))
    for i, r in enumerate(sorted_by_lpips[:3], 1):
        lpips_min = min(r.get('lpips_ref1', 1), r.get('lpips_ref2', 1))
        print(f"   {i}. {r['scheduler']}: LPIPS={lpips_min:.3f}")
    
    print("\n3. BEST BY BLEND QUALITY (highest blend ratio):")
    sorted_by_blend = sorted(successful, key=lambda x: x.get('blend_ratio', 0), reverse=True)
    for i, r in enumerate(sorted_by_blend[:3], 1):
        print(f"   {i}. {r['scheduler']}: {r.get('blend_ratio', 0):.1f}%")
    
    print("\n4. BEST BY LOW ARTIFACTS (minimal HF change):")
    sorted_by_hf = sorted(successful, key=lambda x: abs(x.get('hf_change', 1)))
    for i, r in enumerate(sorted_by_hf[:3], 1):
        print(f"   {i}. {r['scheduler']}: |HF|={abs(r.get('hf_change', 0)):.4f}")
    
    # Overall recommendation
    print("\n" + "="*60)
    print("OVERALL RECOMMENDATIONS")
    print("="*60)
    
    # Score each scheduler
    scheduler_scores = {}
    for r in successful:
        score = 0
        # Lower HF change is better
        score += 1 / (1 + abs(r.get('hf_change', 1)))
        # Higher blend ratio is better
        score += r.get('blend_ratio', 0) / 100
        # Lower LPIPS is better
        score += 1 / (1 + min(r.get('lpips_ref1', 1), r.get('lpips_ref2', 1)))
        # Balanced CLIP is better (close to 0 delta)
        score += 1 / (1 + abs(r.get('clip_delta', 1)))
        
        scheduler_scores[r['scheduler']] = score
    
    # Sort by overall score
    sorted_scores = sorted(scheduler_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTOP 3 OVERALL BEST SCHEDULERS:")
    for i, (sched, score) in enumerate(sorted_scores[:3], 1):
        print(f"   {i}. {sched} (score: {score:.3f})")

print("\n💡 Note: Early prompt switching (step 10) is challenging for all schedulers.")
print("   The best schedulers balance multiple factors: artifacts, perceptual quality,")
print("   prompt alignment, and smooth blending between concepts.")
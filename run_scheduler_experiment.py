#!/usr/bin/env python3
"""Run scheduler experiment with Modal."""
import subprocess
import sys

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
print(f"Running scheduler experiment with prompt switch at step {switch_step}")
print(f"Prompt 1: {prompt_1}")
print(f"Prompt 2: {prompt_2}")
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
            # Parse output to extract HF change
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if "High freq change:" in line:
                    hf_change = float(line.split(":")[1].strip())
                    results.append({
                        "scheduler": scheduler,
                        "hf_change": hf_change,
                        "status": "success"
                    })
                    
                    # Artifact assessment
                    if abs(hf_change) > 0.1:
                        print(f"  ⚠️  Strong artifacts detected! HF change: {hf_change:+.4f}")
                    elif abs(hf_change) > 0.05:
                        print(f"  ⚡ Moderate artifacts detected. HF change: {hf_change:+.4f}")
                    else:
                        print(f"  ✅ Clean transition. HF change: {hf_change:+.4f}")
                    break
        else:
            print(f"❌ Failed with {scheduler}")
            print(f"Error: {result.stderr}")
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

# Summary
print(f"\n{'='*60}")
print("EXPERIMENT SUMMARY")
print(f"{'='*60}")

successful = [r for r in results if r.get("status") == "success"]
print(f"Total schedulers tested: {len(schedulers)}")
print(f"Successful runs: {len(successful)}")
print(f"Failed runs: {len(results) - len(successful)}")

if successful:
    # Sort by HF change magnitude
    sorted_results = sorted(successful, key=lambda x: abs(x['hf_change']))
    
    print(f"\nBest schedulers (least artifacts):")
    for i, r in enumerate(sorted_results[:3], 1):
        print(f"  {i}. {r['scheduler']}: {r['hf_change']:+.4f}")
    
    print(f"\nWorst schedulers (most artifacts):")
    for i, r in enumerate(sorted_results[-3:], 1):
        print(f"  {i}. {r['scheduler']}: {r['hf_change']:+.4f}")
    
    # Group by artifact severity
    strong = [r for r in successful if abs(r['hf_change']) > 0.1]
    moderate = [r for r in successful if 0.05 < abs(r['hf_change']) <= 0.1]
    clean = [r for r in successful if abs(r['hf_change']) <= 0.05]
    
    print(f"\nArtifact severity breakdown:")
    print(f"  Strong artifacts (|HF| > 0.1):   {len(strong)} schedulers")
    print(f"  Moderate artifacts (0.05-0.1):   {len(moderate)} schedulers")
    print(f"  Clean transitions (|HF| ≤ 0.05): {len(clean)} schedulers")
    
    if clean:
        print(f"\n✅ Recommended schedulers for early prompt switching:")
        for r in clean:
            print(f"   - {r['scheduler']}")

print("\n💡 Note: Early prompt switching (step 10) affects high-level image structure.")
print("   Schedulers that handle large noise levels better tend to produce fewer artifacts.")
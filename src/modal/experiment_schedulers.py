"""Experiment to test prompt switching at step 10 with all available schedulers."""
import modal
import json
from pathlib import Path
from typing import List, Dict
import datetime

# Import the modal app
from src.modal.sd3_modal import SD3MechInterp, app


def run_scheduler_experiment(
    prompt_1: str = "A serene mountain landscape with snow peaks",
    prompt_2: str = "A bustling city street at night with neon lights",
    switch_step: int = 10,
    seed: int = 42,
    num_inference_steps: int = 50,
) -> List[Dict]:
    """Run prompt switching experiment with all available schedulers."""
    
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
    
    # Initialize SD3 instance
    sd3 = SD3MechInterp()
    
    results = []
    
    print(f"\n{'='*60}")
    print(f"Running scheduler experiment with prompt switch at step {switch_step}")
    print(f"Prompt 1: {prompt_1}")
    print(f"Prompt 2: {prompt_2}")
    print(f"Total inference steps: {num_inference_steps}")
    print(f"{'='*60}\n")
    
    # Run experiment for each scheduler
    for i, scheduler in enumerate(schedulers, 1):
        print(f"\n[{i}/{len(schedulers)}] Testing scheduler: {scheduler}")
        print("-" * 40)
        
        try:
            # Generate image with prompt switching
            result = sd3.generate_with_prompt_switch.remote(
                prompt_1=prompt_1,
                prompt_2=prompt_2,
                switch_step=switch_step,
                seed=seed,
                num_inference_steps=num_inference_steps,
                scheduler=scheduler,
            )
            
            print(f"✓ Generation complete: {result['artifact_dir']}")
            
            # Run spectral analysis
            spectral_results = sd3.run_basic_spectral_analysis.remote(result['artifact_dir'])
            
            # Extract high frequency change
            summary = spectral_results['analysis_summary']
            hf_change = summary['post_switch_high_freq'] - summary['pre_switch_high_freq']
            
            # Store results
            experiment_result = {
                "scheduler": scheduler,
                "switch_step": switch_step,
                "artifact_dir": result['artifact_dir'],
                "image_path": result['image_path'],
                "num_latents": result['num_latents_captured'],
                "pre_switch_hf": summary['pre_switch_high_freq'],
                "post_switch_hf": summary['post_switch_high_freq'],
                "hf_change": hf_change,
                "spectral_plot": spectral_results['plot_path'],
                "timestamp": result['metadata']['timestamp'],
            }
            
            results.append(experiment_result)
            
            print(f"  Pre-switch HF:  {summary['pre_switch_high_freq']:.4f}")
            print(f"  Post-switch HF: {summary['post_switch_high_freq']:.4f}") 
            print(f"  HF Change:      {hf_change:+.4f}")
            
            # Artifact assessment
            if abs(hf_change) > 0.1:
                print(f"  ⚠️  Strong artifacts detected!")
            elif abs(hf_change) > 0.05:
                print(f"  ⚡ Moderate artifacts detected")
            else:
                print(f"  ✅ Clean transition")
                
        except Exception as e:
            print(f"❌ Error with {scheduler}: {str(e)}")
            results.append({
                "scheduler": scheduler,
                "switch_step": switch_step,
                "error": str(e),
                "hf_change": None,
            })
    
    # Save experiment results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(f"/artifacts/scheduler_experiment_{timestamp}.json")
    
    with open(results_path, "w") as f:
        json.dump({
            "experiment": "scheduler_comparison",
            "switch_step": switch_step,
            "prompt_1": prompt_1,
            "prompt_2": prompt_2,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "timestamp": timestamp,
            "results": results,
        }, f, indent=2)
    
    print(f"\nExperiment results saved to: {results_path}")
    
    # Create comparison visualization
    print("\nCreating comparison visualization...")
    
    # Filter out failed experiments
    valid_results = [r for r in results if r.get('hf_change') is not None]
    
    if valid_results:
        # Prepare data for visualization
        viz_data = []
        for r in valid_results:
            viz_data.append({
                'switch_step': r['switch_step'],
                'hf_change': r['hf_change'],
                'scheduler': r['scheduler'],
                'timestamp': r.get('timestamp', timestamp),
            })
        
        comparison = sd3.create_switch_comparison.remote(viz_data)
        print(f"Comparison plot saved to: {comparison['plot_path']}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total schedulers tested: {len(schedulers)}")
    print(f"Successful runs: {len(valid_results)}")
    print(f"Failed runs: {len(results) - len(valid_results)}")
    
    if valid_results:
        # Sort by HF change magnitude
        sorted_results = sorted(valid_results, key=lambda x: abs(x['hf_change']))
        
        print(f"\nBest schedulers (least artifacts):")
        for i, r in enumerate(sorted_results[:3], 1):
            print(f"  {i}. {r['scheduler']}: {r['hf_change']:+.4f}")
        
        print(f"\nWorst schedulers (most artifacts):")
        for i, r in enumerate(sorted_results[-3:], 1):
            print(f"  {i}. {r['scheduler']}: {r['hf_change']:+.4f}")
    
    return results


@app.local_entrypoint()
def main():
    """Run the scheduler experiment."""
    results = run_scheduler_experiment(
        prompt_1="A serene mountain landscape with snow peaks",
        prompt_2="A bustling city street at night with neon lights",
        switch_step=10,
        seed=42,
        num_inference_steps=50,
    )
    
    # Additional analysis
    print("\n" + "="*60)
    print("DETAILED ANALYSIS")
    print("="*60)
    
    valid_results = [r for r in results if r.get('hf_change') is not None]
    
    if valid_results:
        # Group by artifact severity
        strong_artifacts = [r for r in valid_results if abs(r['hf_change']) > 0.1]
        moderate_artifacts = [r for r in valid_results if 0.05 < abs(r['hf_change']) <= 0.1]
        clean_transitions = [r for r in valid_results if abs(r['hf_change']) <= 0.05]
        
        print(f"\nArtifact severity breakdown:")
        print(f"  Strong artifacts (|HF| > 0.1):   {len(strong_artifacts)} schedulers")
        if strong_artifacts:
            for r in strong_artifacts:
                print(f"    - {r['scheduler']}: {r['hf_change']:+.4f}")
        
        print(f"\n  Moderate artifacts (0.05 < |HF| ≤ 0.1): {len(moderate_artifacts)} schedulers")
        if moderate_artifacts:
            for r in moderate_artifacts:
                print(f"    - {r['scheduler']}: {r['hf_change']:+.4f}")
        
        print(f"\n  Clean transitions (|HF| ≤ 0.05): {len(clean_transitions)} schedulers")
        if clean_transitions:
            for r in clean_transitions:
                print(f"    - {r['scheduler']}: {r['hf_change']:+.4f}")
        
        # Recommendations
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        if clean_transitions:
            print(f"\n✅ Best schedulers for early prompt switching (step {10}):")
            for r in sorted(clean_transitions, key=lambda x: abs(x['hf_change']))[:3]:
                print(f"   - {r['scheduler']}")
        
        print(f"\n⚠️  Schedulers to avoid for early prompt switching:")
        worst = sorted(valid_results, key=lambda x: abs(x['hf_change']), reverse=True)[:3]
        for r in worst:
            print(f"   - {r['scheduler']}")
        
        print("\n💡 Note: Early prompt switching (step 10) affects high-level image structure,")
        print("   so schedulers that handle large noise levels poorly may produce more artifacts.")


if __name__ == "__main__":
    main()
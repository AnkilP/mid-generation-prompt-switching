#!/usr/bin/env python
"""CLI for prompt switching experiments on Modal."""

import modal
import json
from pathlib import Path
from datetime import datetime
from src.modal.sd3_modal import app

@app.local_entrypoint()
def cli(
    prompt1: str = "A serene mountain lake",
    prompt2: str = "A stormy ocean", 
    switch: int = 20,
    preset: str = None
):
    """Run prompt switching experiments on Modal."""
    
    # Check if preset mode
    if preset:
        run_preset(preset)
        return
    
    # Single experiment mode
    print(f"🚀 Running single experiment on Modal")
    print(f"   From: {prompt1}")
    print(f"   To:   {prompt2}")
    print(f"   Switch: step {switch}")
    print("-" * 60)
    
    from src.modal.sd3_modal import SD3MechInterp
    
    sd3 = SD3MechInterp()
    
    result = sd3.generate_with_prompt_switch.remote(
        prompt_1=prompt1,
        prompt_2=prompt2,
        switch_step=switch,
        seed=42,
        num_inference_steps=50,
        guidance_scale=7.0,
        capture_every_n_steps=5,
    )
    
    # Save image locally
    from pathlib import Path
    
    local_dir = Path("data/images")
    local_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = result['metadata']['timestamp']
    local_file = local_dir / f"{timestamp}_prompt_switch.png"
    
    # Write image bytes to local file
    if 'image_bytes' in result:
        with open(local_file, 'wb') as f:
            f.write(result['image_bytes'])
        print(f"📥 Image saved locally: {local_file}")
    
    print(f"\n✅ Success!")
    print(f"   Remote Image: {result['image_path']}")
    print(f"   Local Image: {local_file}")
    print(f"   Artifacts: {result['artifact_dir']}")
    
    # Analysis
    spectral = sd3.run_basic_spectral_analysis.remote(result['artifact_dir'])
    change = spectral['analysis_summary']['post_switch_high_freq'] - \
             spectral['analysis_summary']['pre_switch_high_freq']
    
    print(f"\n📊 High frequency change: {change:+.4f}")
    
    if abs(change) > 0.1:
        print("   ⚠️  Strong artifact!")
    elif abs(change) > 0.05:
        print("   📌 Moderate artifact")
    else:
        print("   ✨ Minimal artifact")


def run_preset(preset_name: str):
    """Run a batch of experiments from preset."""
    presets = {
        'quick': {
            'prompts': [
                ("A photo of a cat", "A photo of a dog"),
                ("Day scene", "Night scene"),
            ],
            'switch_steps': [10, 20, 30, 40],
            'seeds': [42]
        },
        'contrast': {
            'prompts': [
                ("Peaceful zen garden", "Chaotic battlefield"),
                ("Smooth glass", "Rough bark"),
                ("Silent library", "Rock concert"),
            ],
            'switch_steps': [10, 20, 30],
            'seeds': [42, 123]
        },
        'style': {
            'prompts': [
                ("Mountain photo", "Mountain painting"),
                ("Cat realistic", "Cat cartoon"),
            ],
            'switch_steps': [15, 25, 35],
            'seeds': [42]
        },
        'step_analysis': {
            'prompts': [
                ("A lion", "A tiger"),
            ],
            'switch_steps': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45],
            'seeds': [42]
        }
    }
    
    if preset_name not in presets:
        print(f"Unknown preset: {preset_name}")
        print(f"Available: {list(presets.keys())}")
        return
    
    config = presets[preset_name]
    total = len(config['prompts']) * len(config['switch_steps']) * len(config['seeds'])
    
    print(f"🚀 Running {preset_name} preset: {total} experiments")
    print("-" * 60)
    
    from src.modal.sd3_modal import SD3MechInterp
    sd3 = SD3MechInterp()
    
    results = []
    for i, (p1, p2) in enumerate(config['prompts']):
        for switch_step in config['switch_steps']:
            for seed in config['seeds']:
                print(f"\n[{len(results)+1}/{total}] {p1[:20]}... → {p2[:20]}... @ {switch_step}")
                
                try:
                    result = sd3.generate_with_prompt_switch.remote(
                        prompt_1=p1,
                        prompt_2=p2,
                        switch_step=switch_step,
                        seed=seed,
                        num_inference_steps=50,
                        guidance_scale=7.0,
                        capture_every_n_steps=5,
                    )
                    
                    spectral = sd3.run_basic_spectral_analysis.remote(result['artifact_dir'])
                    hf_change = spectral['analysis_summary']['post_switch_high_freq'] - \
                               spectral['analysis_summary']['pre_switch_high_freq']
                    
                    print(f"   ✅ HF change: {hf_change:+.4f}")
                    # Save image locally for this experiment
                    timestamp = result['metadata']['timestamp']
                    local_dir = Path("data/images")
                    local_dir.mkdir(parents=True, exist_ok=True)
                    local_file = local_dir / f"{timestamp}_{switch_step}_switch.png"
                    
                    if 'image_bytes' in result:
                        with open(local_file, 'wb') as f:
                            f.write(result['image_bytes'])
                    
                    results.append({
                        'success': True,
                        'prompt_1': p1,
                        'prompt_2': p2,
                        'switch_step': switch_step,
                        'seed': seed,
                        'hf_change': hf_change,
                        'artifact_dir': result['artifact_dir'],
                        'local_image': str(local_file),
                        'timestamp': timestamp
                    })
                    
                except Exception as e:
                    print(f"   ❌ Failed: {e}")
                    results.append({'success': False, 'error': str(e)})
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"data/experiments/{preset_name}_{timestamp}_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    successful = sum(1 for r in results if r.get('success'))
    successful_results = [r for r in results if r.get('success')]
    
    print(f"\n{'='*60}")
    print(f"✅ Complete: {successful}/{total} successful")
    print(f"💾 Results: {output_file}")
    
    # Create switch step comparison visualization
    if successful_results:
        print("\n📊 Creating switch step comparison...")
        comparison = sd3.create_switch_comparison.remote(successful_results)
        
        if 'plot_path' in comparison:
            print(f"📈 Comparison plot: {comparison['plot_path']}")
            
            # Print analysis summary
            analysis = comparison['analysis']
            print(f"\n🔍 Analysis Summary:")
            print(f"   • Strongest artifact: {analysis['strongest_artifact']:+.4f} at step {analysis['strongest_step']}")
            print(f"   • Strong artifacts: {analysis['strong_artifacts']}")
            print(f"   • Moderate artifacts: {analysis['moderate_artifacts']}")
            print(f"   • Clean generations: {analysis['clean_generations']}")
            print(f"   • {analysis['interpretation']}")


@app.local_entrypoint()
def preset_quick():
    """Run quick preset experiments."""
    run_preset("quick")

@app.local_entrypoint()
def preset_step_analysis():
    """Run step analysis preset experiments."""
    run_preset("step_analysis")

@app.local_entrypoint()
def preset_contrast():
    """Run contrast preset experiments."""
    run_preset("contrast")

@app.local_entrypoint()
def preset_style():
    """Run style preset experiments."""
    run_preset("style")


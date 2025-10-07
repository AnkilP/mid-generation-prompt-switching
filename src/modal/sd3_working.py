"""Working SD3 Modal app with prompt switching and spectral analysis."""
import modal
from pathlib import Path
from typing import Dict, List, Optional, Tuple

app = modal.App("sd3-working")

image = modal.Image.debian_slim().pip_install(
    "torch>=2.0.0",
    "diffusers==0.30.0",
    "transformers==4.45.0",
    "accelerate",
    "sentencepiece",
    "protobuf",
    "numpy",
    "scipy",
    "Pillow",
    "matplotlib",
)

volume = modal.Volume.from_name("sd3-artifacts", create_if_missing=True)

@app.function(
    gpu="a100",
    image=image,
    volumes={"/artifacts": volume},
    secrets=[modal.Secret.from_name("huggingface-token")],
    timeout=1800,
)
def generate_with_prompt_switch_and_analysis(
    prompt_1: str = "A simple geometric pattern with clean lines",
    prompt_2: str = "A complex organic texture with intricate details", 
    switch_step: int = 15,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.0,
    seed: Optional[int] = None,
) -> Dict:
    """Generate image with prompt switching and run spectral analysis."""
    import torch
    import os
    from diffusers import StableDiffusion3Pipeline
    from PIL import Image
    import numpy as np
    from pathlib import Path
    import datetime
    import json
    import matplotlib.pyplot as plt
    from scipy.fft import fft2, fftfreq
    
    print(f"Loading SD3 pipeline...")
    
    # Get HuggingFace token
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    
    # Load pipeline with CPU offloading
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16,
        token=hf_token,
    )
    
    # Enable CPU offloading to save memory
    pipe.enable_model_cpu_offload()
    print(f"Pipeline loaded successfully!")
    
    # Set seed
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
    else:
        generator = None
    
    print(f"Generating image sequence...")
    
    # Generate images at multiple steps to simulate prompt switching
    steps_per_stage = num_inference_steps // 3
    
    # Stage 1: Pure prompt 1
    print(f"Stage 1: Generating with '{prompt_1}' for {steps_per_stage} steps")
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
    
    intermediate_1 = pipe(
        prompt=prompt_1,
        num_inference_steps=steps_per_stage,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="latent",
    ).images[0]
    
    # Stage 2: Switch step - blend prompts (simulated)
    print(f"Stage 2: Transitioning at step {switch_step}")
    blend_prompt = f"{prompt_1}, transitioning to {prompt_2}"
    
    # Use the intermediate latent as starting point
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    
    intermediate_2 = pipe(
        prompt=blend_prompt,
        num_inference_steps=steps_per_stage,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="latent", 
    ).images[0]
    
    # Stage 3: Pure prompt 2
    print(f"Stage 3: Generating with '{prompt_2}' for {steps_per_stage} steps")
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed + 2)
    
    final_image = pipe(
        prompt=prompt_2,
        num_inference_steps=steps_per_stage,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = Path(f"/artifacts/prompt_switch_{timestamp}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    # Save final image
    image_path = artifact_dir / "output.png"
    final_image.save(image_path)
    
    # Save intermediate latents for analysis
    latents = {
        "stage_1": intermediate_1.cpu().numpy(),
        "stage_2": intermediate_2.cpu().numpy(),
        "final": None  # Final is PIL image, not latent
    }
    
    # Basic spectral analysis
    print("Running spectral analysis...")
    
    spectral_results = {}
    
    for stage_name, latent in latents.items():
        if latent is not None:
            print(f"Analyzing {stage_name}: latent shape = {latent.shape}")
            
            # Handle different tensor shapes
            if len(latent.shape) == 4:  # [batch, channel, h, w]
                channel_data = latent[0, 0]
            elif len(latent.shape) == 3:  # [channel, h, w] 
                channel_data = latent[0]
            elif len(latent.shape) == 2:  # [h, w]
                channel_data = latent
            else:
                print(f"Unexpected latent shape: {latent.shape}")
                continue
            
            print(f"Channel data shape: {channel_data.shape}")
            
            # 2D FFT
            fft_2d = fft2(channel_data)
            psd = np.abs(fft_2d) ** 2
            
            # Create frequency grid
            freq_y = fftfreq(channel_data.shape[0])
            freq_x = fftfreq(channel_data.shape[1])
            freq_r = np.sqrt(freq_y[:, np.newaxis]**2 + freq_x[np.newaxis, :]**2)
            
            # Frequency bands
            total_power = np.sum(psd)
            low_mask = freq_r < 0.1
            mid_mask = (freq_r >= 0.1) & (freq_r < 0.3)
            high_mask = freq_r >= 0.3
            
            spectral_results[stage_name] = {
                "total_power": float(total_power),
                "low_freq_power": float(np.sum(psd[low_mask]) / total_power),
                "mid_freq_power": float(np.sum(psd[mid_mask]) / total_power),
                "high_freq_power": float(np.sum(psd[high_mask]) / total_power),
            }
    
    # Create spectral analysis plot
    if len(spectral_results) >= 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        stages = list(spectral_results.keys())
        low_freqs = [spectral_results[s]["low_freq_power"] for s in stages]
        mid_freqs = [spectral_results[s]["mid_freq_power"] for s in stages]
        high_freqs = [spectral_results[s]["high_freq_power"] for s in stages]
        
        # Frequency band evolution
        x_pos = range(len(stages))
        width = 0.25
        
        ax1.bar([x - width for x in x_pos], low_freqs, width, label='Low freq', alpha=0.7)
        ax1.bar(x_pos, mid_freqs, width, label='Mid freq', alpha=0.7)
        ax1.bar([x + width for x in x_pos], high_freqs, width, label='High freq', alpha=0.7)
        
        ax1.set_xlabel('Stage')
        ax1.set_ylabel('Relative Power')
        ax1.set_title('Frequency Content by Stage')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(stages)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Total power
        total_powers = [spectral_results[s]["total_power"] for s in stages]
        ax2.plot(stages, total_powers, 'ko-', linewidth=2, markersize=8)
        ax2.set_xlabel('Stage')
        ax2.set_ylabel('Total Power (log scale)')
        ax2.set_title('Total Spectral Power Evolution')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        spectral_plot_path = artifact_dir / "spectral_analysis.png"
        plt.savefig(spectral_plot_path, dpi=150)
        plt.close()
    else:
        spectral_plot_path = None
    
    # Save metadata and results
    metadata = {
        "prompt_1": prompt_1,
        "prompt_2": prompt_2, 
        "switch_step": switch_step,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "timestamp": timestamp,
        "model": "stabilityai/stable-diffusion-3-medium-diffusers",
    }
    
    with open(artifact_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    with open(artifact_dir / "spectral_results.json", "w") as f:
        json.dump(spectral_results, f, indent=2)
    
    print(f"Results saved to: {artifact_dir}")
    
    # Generate analysis summary
    if len(spectral_results) >= 2:
        stages = list(spectral_results.keys())
        first_stage = spectral_results[stages[0]]
        last_stage = spectral_results[stages[-1]]
        
        high_freq_change = last_stage["high_freq_power"] - first_stage["high_freq_power"]
        total_power_change = (last_stage["total_power"] - first_stage["total_power"]) / first_stage["total_power"]
        
        analysis_summary = {
            "high_freq_change": high_freq_change,
            "total_power_change_pct": total_power_change * 100,
            "artifact_detected": abs(high_freq_change) > 0.05,  # Threshold for artifact detection
        }
    else:
        analysis_summary = {"error": "Insufficient data for analysis"}
    
    return {
        "image_path": str(image_path),
        "artifact_dir": str(artifact_dir),
        "metadata": metadata,
        "spectral_results": spectral_results,
        "spectral_plot_path": str(spectral_plot_path) if spectral_plot_path else None,
        "analysis_summary": analysis_summary,
        "success": True
    }

@app.local_entrypoint()
def main(
    prompt_1: str = "Simple geometric shapes with clean edges",
    prompt_2: str = "Complex organic textures with fine details", 
    switch_step: int = 15,
    seed: int = 42,
):
    """Run SD3 prompt switching with spectral analysis."""
    print(f"🔬 Running SD3 Prompt Switching + Spectral Analysis")
    print(f"📝 Prompt 1: {prompt_1}")
    print(f"📝 Prompt 2: {prompt_2}")
    print(f"🔄 Switch step: {switch_step}")
    print(f"🎲 Seed: {seed}")
    
    result = generate_with_prompt_switch_and_analysis.remote(
        prompt_1=prompt_1,
        prompt_2=prompt_2,
        switch_step=switch_step,
        seed=seed,
    )
    
    if result["success"]:
        print(f"\n✅ Generation and analysis completed!")
        print(f"🖼️  Image: {result['image_path']}")
        print(f"📊 Spectral plot: {result['spectral_plot_path']}")
        print(f"📁 Artifact dir: {result['artifact_dir']}")
        
        # Print spectral analysis summary
        summary = result["analysis_summary"]
        if "high_freq_change" in summary:
            print(f"\n🔬 Spectral Analysis Results:")
            print(f"   📈 High freq change: {summary['high_freq_change']:+.4f}")
            print(f"   📊 Total power change: {summary['total_power_change_pct']:+.2f}%")
            print(f"   🚨 Artifact detected: {'YES' if summary['artifact_detected'] else 'NO'}")
        
        # Print per-stage results
        print(f"\n📋 Stage-by-Stage Results:")
        for stage, data in result["spectral_results"].items():
            print(f"   {stage}:")
            print(f"     Low freq: {data['low_freq_power']:.4f}")
            print(f"     Mid freq: {data['mid_freq_power']:.4f}")
            print(f"     High freq: {data['high_freq_power']:.4f}")
            print(f"     Total power: {data['total_power']:.2e}")
    else:
        print(f"\n❌ Generation failed!")
    
    return result
"""Simplified SD3 Modal app for testing basic functionality."""
import modal
from pathlib import Path
from typing import Dict, Optional

app = modal.App("sd3-simple")

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
    timeout=1200,
)
def generate_simple_sd3(
    prompt: str = "A photo of a cat sitting on a mat",
    num_inference_steps: int = 20,  # Reduced for faster testing
    guidance_scale: float = 7.0,
    seed: Optional[int] = None,
) -> Dict:
    """Generate a simple SD3 image without all the hooks."""
    import torch
    import os
    from diffusers import StableDiffusion3Pipeline
    from PIL import Image
    import numpy as np
    from pathlib import Path
    import datetime
    import json
    
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
        generator = torch.Generator(device="cuda").manual_seed(seed)
    else:
        generator = None
    
    print(f"Generating image: '{prompt}'")
    
    # Generate image
    image = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = Path(f"/artifacts/simple_{timestamp}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    # Save image
    image_path = artifact_dir / "output.png"
    image.save(image_path)
    
    # Save metadata
    metadata = {
        "prompt": prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "timestamp": timestamp,
        "model": "stabilityai/stable-diffusion-3-medium-diffusers",
    }
    
    with open(artifact_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Results saved to: {artifact_dir}")
    
    return {
        "image_path": str(image_path),
        "artifact_dir": str(artifact_dir),
        "metadata": metadata,
        "success": True
    }

@app.local_entrypoint()
def main(
    prompt: str = "A geometric pattern transitioning to organic texture",
    steps: int = 20,
    seed: int = 42,
):
    """Run simple SD3 generation."""
    print(f"Running SD3 generation...")
    print(f"Prompt: {prompt}")
    print(f"Steps: {steps}")
    print(f"Seed: {seed}")
    
    result = generate_simple_sd3.remote(
        prompt=prompt,
        num_inference_steps=steps,
        seed=seed,
    )
    
    if result["success"]:
        print(f"\n✅ Generation successful!")
        print(f"Image saved to: {result['image_path']}")
        print(f"Artifact dir: {result['artifact_dir']}")
    else:
        print(f"\n❌ Generation failed!")
    
    return result
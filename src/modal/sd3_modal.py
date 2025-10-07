"""Modal app for SD3 with mid-generation prompt switching and latent caching."""
import modal
from pathlib import Path
from typing import Dict, List, Optional, Tuple

app = modal.App("sd3-mech-interp")

# Stable image that should work
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "diffusers==0.31.0",
        "transformers==4.47.0", 
        "accelerate==1.1.1",
        "sentencepiece",
        "numpy",
        "scipy",
        "Pillow",
        "matplotlib",
        "open-clip-torch",  # For CLIP models
        "lpips",  # For perceptual distance metrics
        "torchmetrics",  # For additional metrics
        "torchvision",  # For image transforms
    )
)

volume = modal.Volume.from_name("sd3-artifacts", create_if_missing=True)


@app.cls(
    gpu="h100",
    image=image,
    volumes={"/artifacts": volume},
    secrets=[modal.Secret.from_name("huggingface-token")],
    timeout=1200,
)
class SD3MechInterp:
        
    @modal.enter()
    def setup(self):
        import torch
        import os
        from diffusers import StableDiffusion3Pipeline
        
        # Initialize attributes that were in __init__
        self.device = "cuda"
        self.dtype = torch.float16
        self.latent_cache = {}
        self.attention_maps = {}
        
        # Get HuggingFace token from environment
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        
        # Load SD3 pipeline with minimal parameters to avoid compatibility issues
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=self.dtype,
            token=hf_token,
        )
        
        # Manually move to GPU
        self.pipe = self.pipe.to(self.device)
        
        # Load CLIP model for image-text similarity
        import open_clip
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai'
        )
        self.clip_model = self.clip_model.to(self.device)
        self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
        
        # Load LPIPS for perceptual distance
        import lpips
        self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
        
        self.setup_hooks()  # Enable to capture attention maps
    
    def setup_hooks(self):
        """Set up hooks to capture intermediate latents and attention maps."""
        self.hook_handles = []
        self.attention_weights = {}  # Store attention weights separately
        
        def create_latent_hook(name):
            def hook(module, input, output):
                if hasattr(self, 'step_counter') and output is not None:
                    key = f"{name}_step_{self.step_counter}"
                    if isinstance(output, tuple):
                        output = output[0]
                    if output is not None and hasattr(output, 'detach'):
                        self.latent_cache[key] = output.detach().cpu()
            return hook
        
        def create_attention_hook(name):
            def hook(module, input, output):
                if hasattr(self, 'step_counter'):
                    key = f"{name}_step_{self.step_counter}"
                    
                    # Try different ways to capture attention
                    attention_tensor = None
                    
                    # Method 1: Check for attention_probs attribute
                    if hasattr(module, 'attention_probs') and module.attention_probs is not None:
                        attention_tensor = module.attention_probs
                    
                    # Method 2: Check if output contains attention weights
                    elif isinstance(output, tuple) and len(output) > 1:
                        # Second element might be attention weights
                        attention_tensor = output[1]
                    
                    # Method 3: Look for attention weights in module state
                    elif hasattr(module, 'attn_weights') and module.attn_weights is not None:
                        attention_tensor = module.attn_weights
                    
                    # Method 4: Try to extract attention from input tensors
                    elif len(input) > 0 and hasattr(input[0], 'shape'):
                        # Some attention modules store weights in intermediate calculations
                        # For now, use input tensor statistics as a proxy
                        import torch
                        query_tensor = input[0]
                        if len(query_tensor.shape) >= 3:  # [batch, seq, hidden]
                            # Compute attention-like statistics from the query tensor
                            attention_proxy = torch.std(query_tensor, dim=-1)  # [batch, seq]
                            attention_tensor = attention_proxy
                    
                    # Method 5: Store debug info if nothing found
                    else:
                        # Store some debug info about the module
                        self.attention_weights[key] = {
                            'module_type': type(module).__name__,
                            'has_attention_probs': hasattr(module, 'attention_probs'),
                            'has_attn_weights': hasattr(module, 'attn_weights'),
                            'output_type': type(output).__name__ if output is not None else 'None',
                            'output_len': len(output) if isinstance(output, (tuple, list)) else 'not_sequence',
                            'input_shapes': [inp.shape if hasattr(inp, 'shape') else 'no_shape' for inp in input] if input else [],
                        }
                    
                    if attention_tensor is not None and hasattr(attention_tensor, 'detach'):
                        # Debug: print info about captured attention tensors
                        if hasattr(self, 'step_counter') and self.step_counter <= 15 and len(self.attention_maps) < 10:
                            tensor_shape = attention_tensor.shape if hasattr(attention_tensor, 'shape') else 'no_shape'
                            tensor_dtype = attention_tensor.dtype if hasattr(attention_tensor, 'dtype') else 'no_dtype'
                            print(f"Captured attention at step {self.step_counter}: {key}, shape={tensor_shape}, dtype={tensor_dtype}")
                        
                        self.attention_maps[key] = attention_tensor.detach().cpu()
            return hook
        
        # Hook into transformer blocks
        for idx, block in enumerate(self.pipe.transformer.transformer_blocks):
            handle = block.register_forward_hook(create_latent_hook(f"block_{idx}"))
            self.hook_handles.append(handle)
            
            # Hook different attention components
            # Try to find attention modules in the block
            for name, module in block.named_modules():
                if any(keyword in name.lower() for keyword in ['attn', 'attention']):
                    attn_handle = module.register_forward_hook(
                        create_attention_hook(f"block_{idx}_{name}")
                    )
                    self.hook_handles.append(attn_handle)
    
    def cleanup_hooks(self):
        """Remove all hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
    
    def compute_clip_scores(self, image, prompt_1: str, prompt_2: str) -> Dict[str, float]:
        """Compute CLIP similarity scores between image and both prompts."""
        import torch
        import open_clip
        from PIL import Image
        
        # Preprocess image
        if isinstance(image, Image.Image):
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        else:
            # If tensor, convert to PIL first
            image_pil = Image.fromarray((image * 255).astype('uint8'))
            image_input = self.clip_preprocess(image_pil).unsqueeze(0).to(self.device)
        
        # Tokenize prompts
        text_tokens_1 = self.clip_tokenizer([prompt_1]).to(self.device)
        text_tokens_2 = self.clip_tokenizer([prompt_2]).to(self.device)
        
        # Compute features
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features_1 = self.clip_model.encode_text(text_tokens_1)
            text_features_2 = self.clip_model.encode_text(text_tokens_2)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features_1 = text_features_1 / text_features_1.norm(dim=-1, keepdim=True)
            text_features_2 = text_features_2 / text_features_2.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarities
            similarity_1 = (image_features @ text_features_1.T).squeeze().item()
            similarity_2 = (image_features @ text_features_2.T).squeeze().item()
        
        return {
            "clip_score_prompt_1": similarity_1,
            "clip_score_prompt_2": similarity_2,
            "clip_score_delta": abs(similarity_2 - similarity_1),  # Absolute difference - measures semantic change magnitude
        }
    
    def compute_lpips_distance(self, image_1, image_2) -> float:
        """Compute LPIPS perceptual distance between two images."""
        import torch
        from PIL import Image
        import torchvision.transforms as transforms
        
        # Transform to normalize images to [-1, 1] range expected by LPIPS
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Process images
        if isinstance(image_1, Image.Image):
            img1_tensor = transform(image_1).unsqueeze(0).to(self.device)
        else:
            img1_tensor = transform(Image.fromarray((image_1 * 255).astype('uint8'))).unsqueeze(0).to(self.device)
            
        if isinstance(image_2, Image.Image):
            img2_tensor = transform(image_2).unsqueeze(0).to(self.device)
        else:
            img2_tensor = transform(Image.fromarray((image_2 * 255).astype('uint8'))).unsqueeze(0).to(self.device)
        
        # Compute LPIPS distance
        with torch.no_grad():
            distance = self.lpips_model(img1_tensor, img2_tensor).squeeze().item()
        
        return distance
    
    def analyze_attention_variance(self) -> Dict[str, float]:
        """Analyze variance in captured attention maps."""
        import numpy as np
        
        # Check if we have actual attention maps
        if self.attention_maps:
            variances = {}
            
            for key, attn_map in self.attention_maps.items():
                try:
                    # Convert to numpy if tensor
                    if hasattr(attn_map, 'cpu'):
                        attn_map = attn_map.cpu().numpy()
                    elif hasattr(attn_map, 'detach'):
                        attn_map = attn_map.detach().cpu().numpy()
                    
                    # Check for invalid values
                    if not np.isfinite(attn_map).all():
                        # Replace inf/nan with 0
                        attn_map = np.nan_to_num(attn_map, nan=0.0, posinf=1.0, neginf=0.0)
                    
                    # Compute variance across spatial dimensions with overflow protection
                    if len(attn_map.shape) >= 2:
                        # Flatten and compute robust variance
                        flat_attn = attn_map.reshape(-1)
                        
                        # Use float64 for intermediate calculations to prevent overflow
                        flat_attn_64 = flat_attn.astype(np.float64)
                        
                        # Normalize the data to prevent overflow in variance calculation
                        # Use standard deviation of normalized data as a measure of variance
                        attn_min, attn_max = np.min(flat_attn_64), np.max(flat_attn_64)
                        attn_range = attn_max - attn_min
                        
                        if attn_range > 0:
                            # Normalize to [0, 1] range
                            normalized_attn = (flat_attn_64 - attn_min) / attn_range
                            variance = np.var(normalized_attn)
                        else:
                            # All values are the same
                            variance = 0.0
                        
                        # Debug: print step info for first few maps
                        if 'step_' in key and len(variances) < 5:
                            step_num = key.split('step_')[-1].split('_')[0] if 'step_' in key else 'unknown'
                            unique_vals = len(np.unique(flat_attn))
                            print(f"Attention map {key}: norm_variance={variance:.6f}, unique_vals={unique_vals}, range=[{attn_min:.6f}, {attn_max:.6f}], shape={attn_map.shape}")
                        
                        # Ensure variance is finite and reasonable
                        if np.isfinite(variance) and variance >= 0:
                            variances[key] = float(variance)
                        else:
                            if len(variances) < 5:  # Only print for first few
                                print(f"Warning: Invalid variance for {key}: {variance}")
                            variances[key] = 0.0
                    
                except Exception as e:
                    # If anything goes wrong, skip this attention map
                    print(f"Warning: Error processing attention map {key}: {e}")
                    continue
            
            print(f"Attention variance debug: {len(variances)} valid variances from {len(self.attention_maps)} maps")
            if len(variances) > 0:
                sample_variances = list(variances.values())[:5]
                print(f"Sample variances: {sample_variances}")
            
            # Also print some debug info
            if hasattr(self, 'attention_weights') and self.attention_weights:
                print(f"Debug info available for {len(self.attention_weights)} modules")
                sample_debug = list(self.attention_weights.values())[:2]
                for i, debug in enumerate(sample_debug):
                    print(f"  Module {i}: {debug}")
            
            # Compute statistics with safety checks
            all_variances = [v for v in variances.values() if np.isfinite(v)]
            
            if all_variances:
                # Use robust statistics
                mean_var = float(np.mean(all_variances))
                std_var = float(np.std(all_variances))
                max_var = float(np.max(all_variances))
                min_var = float(np.min(all_variances))
                
                # Cap extreme values for safety
                max_reasonable = 1000.0
                if max_var > max_reasonable:
                    print(f"Warning: Capping extreme variance {max_var} to {max_reasonable}")
                    max_var = max_reasonable
                
                return {
                    "attention_variances": {k: min(v, max_reasonable) for k, v in variances.items()},
                    "mean_variance": min(mean_var, max_reasonable),
                    "std_variance": min(std_var, max_reasonable),
                    "max_variance": max_var,
                    "min_variance": min_var,
                    "num_captured": len(all_variances),
                    "num_total_maps": len(self.attention_maps),
                    "num_valid_maps": len(all_variances),
                }
            else:
                return {
                    "error": "No valid attention variances computed",
                    "num_total_maps": len(self.attention_maps),
                    "num_valid_maps": 0,
                }
        
        # If no attention maps, return debug info
        debug_info = {
            "error": "No attention maps captured",
            "num_attention_modules_found": len([k for k in getattr(self, 'attention_weights', {}).keys()]),
            "debug_info": getattr(self, 'attention_weights', {}),
        }
        
        # Try to analyze the debug info to understand the architecture
        if hasattr(self, 'attention_weights') and self.attention_weights:
            module_types = set()
            attrs_found = set()
            
            for info in self.attention_weights.values():
                if isinstance(info, dict):
                    module_types.add(info.get('module_type', 'Unknown'))
                    attrs_found.update(info.get('module_attrs', []))
            
            debug_info.update({
                "module_types_found": list(module_types),
                "attention_attrs_found": list(attrs_found),
                "total_modules_hooked": len(self.attention_weights),
            })
        
        return debug_info
    
    @modal.method()
    def inspect_sd3_architecture(self) -> Dict:
        """Inspect SD3 transformer architecture to understand attention structure."""
        architecture_info = {
            "transformer_type": type(self.pipe.transformer).__name__,
            "num_blocks": len(self.pipe.transformer.transformer_blocks),
            "blocks_info": []
        }
        
        for idx, block in enumerate(self.pipe.transformer.transformer_blocks):
            block_info = {
                "block_idx": idx,
                "block_type": type(block).__name__,
                "modules": {}
            }
            
            # Inspect each module in the block
            for name, module in block.named_modules():
                if name:  # Skip the block itself
                    module_info = {
                        "type": type(module).__name__,
                        "has_attention_attrs": [attr for attr in dir(module) if 'attn' in attr.lower()],
                        "has_weight": hasattr(module, 'weight'),
                        "parameters_count": sum(p.numel() for p in module.parameters()) if hasattr(module, 'parameters') else 0
                    }
                    block_info["modules"][name] = module_info
            
            architecture_info["blocks_info"].append(block_info)
            
            # Only inspect first 2 blocks to avoid too much data
            if idx >= 1:
                break
        
        return architecture_info
    
    @modal.method()
    def generate_with_prompt_switch(
        self,
        prompt_1: str,
        prompt_2: str,
        switch_step: int = 25,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        seed: Optional[int] = None,
        capture_every_n_steps: int = 5,
        scheduler: str = "FlowMatchEulerDiscreteScheduler",
    ) -> Dict:
        """Generate image with prompt switching at specified step."""
        import torch
        from PIL import Image
        import numpy as np
        from diffusers import (
            FlowMatchEulerDiscreteScheduler,
            FlowMatchHeunDiscreteScheduler,
            DEISMultistepScheduler,
            DPMSolverMultistepScheduler,
            DPMSolverSinglestepScheduler,
            KDPM2DiscreteScheduler,
            KDPM2AncestralDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            HeunDiscreteScheduler,
            PNDMScheduler,
            DDIMScheduler,
            DDPMScheduler,
            LCMScheduler,
        )
        
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Reset caches
        self.latent_cache = {}
        self.attention_maps = {}
        self.step_counter = 0
        
        # Configure scheduler based on user choice
        scheduler_map = {
            "FlowMatchEulerDiscreteScheduler": FlowMatchEulerDiscreteScheduler,
            "FlowMatchHeunDiscreteScheduler": FlowMatchHeunDiscreteScheduler,
            "DEISMultistepScheduler": DEISMultistepScheduler,
            "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
            "DPMSolverSinglestepScheduler": DPMSolverSinglestepScheduler,
            "KDPM2DiscreteScheduler": KDPM2DiscreteScheduler,
            "KDPM2AncestralDiscreteScheduler": KDPM2AncestralDiscreteScheduler,
            "EulerDiscreteScheduler": EulerDiscreteScheduler,
            "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
            "HeunDiscreteScheduler": HeunDiscreteScheduler,
            "PNDMScheduler": PNDMScheduler,
            "DDIMScheduler": DDIMScheduler,
            "DDPMScheduler": DDPMScheduler,
            "LCMScheduler": LCMScheduler,
        }
        
        if scheduler not in scheduler_map:
            print(f"Warning: Unknown scheduler '{scheduler}'. Using default FlowMatchEulerDiscreteScheduler.")
            scheduler = "FlowMatchEulerDiscreteScheduler"
        
        # Create scheduler instance with the pipeline's config
        scheduler_class = scheduler_map[scheduler]
        self.pipe.scheduler = scheduler_class.from_config(self.pipe.scheduler.config)
        
        # Proper manual denoising loop with actual prompt switching
        print(f"Starting generation with prompt switching at step {switch_step}")
        print(f"Using scheduler: {scheduler}")
        
        # Encode both prompts first
        print("Encoding prompts...")
        (
            prompt_embeds_1,
            negative_prompt_embeds_1,
            pooled_prompt_embeds_1,
            negative_pooled_prompt_embeds_1,
        ) = self.pipe.encode_prompt(
            prompt=prompt_1,
            prompt_2=None,
            prompt_3=None,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=guidance_scale > 1.0,
            negative_prompt=None,
            negative_prompt_2=None,
            negative_prompt_3=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            clip_skip=None,
        )
        
        (
            prompt_embeds_2,
            negative_prompt_embeds_2,
            pooled_prompt_embeds_2,
            negative_pooled_prompt_embeds_2,
        ) = self.pipe.encode_prompt(
            prompt=prompt_2,
            prompt_2=None,
            prompt_3=None,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=guidance_scale > 1.0,
            negative_prompt=None,
            negative_prompt_2=None,
            negative_prompt_3=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            clip_skip=None,
        )
        
        # Prepare latents
        batch_size = 1
        height = width = 1024
        
        # Get latents shape and create manually
        num_channels_latents = self.pipe.transformer.config.in_channels
        shape = (
            batch_size,
            num_channels_latents,
            height // self.pipe.vae_scale_factor,
            width // self.pipe.vae_scale_factor,
        )
        
        # Create latents manually to avoid device issues
        import torch
        latents = torch.randn(shape, generator=generator, device=torch.device(self.device), dtype=prompt_embeds_1.dtype)
        
        # Prepare timesteps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps
        
        # Manual denoising loop with prompt switching
        print(f"Running {num_inference_steps} denoising steps...")
        
        with torch.no_grad():  # Disable gradient computation to save memory
            for i, t in enumerate(timesteps):
                self.step_counter = i
                
                # Switch prompts at the specified step
                if i < switch_step:
                    prompt_embeds = prompt_embeds_1
                    negative_prompt_embeds = negative_prompt_embeds_1
                    pooled_prompt_embeds = pooled_prompt_embeds_1
                    negative_pooled_prompt_embeds = negative_pooled_prompt_embeds_1
                else:
                    prompt_embeds = prompt_embeds_2
                    negative_prompt_embeds = negative_prompt_embeds_2
                    pooled_prompt_embeds = pooled_prompt_embeds_2
                    negative_pooled_prompt_embeds = negative_pooled_prompt_embeds_2
                    if i == switch_step:
                        print(f"Step {i}: Switching to prompt: '{prompt_2}'")
                
                # Expand the latents if doing classifier free guidance
                if guidance_scale > 1.0:
                    latent_model_input = torch.cat([latents] * 2)
                    prompt_embeds_input = torch.cat([negative_prompt_embeds, prompt_embeds])
                    pooled_prompt_embeds_input = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])
                else:
                    latent_model_input = latents
                    prompt_embeds_input = prompt_embeds
                    pooled_prompt_embeds_input = pooled_prompt_embeds
                
                # Predict the noise residual
                # Ensure timestep is in the correct format
                timestep_tensor = t.unsqueeze(0) if t.dim() == 0 else t
                
                # Enable gradient computation only for the forward pass we need
                with torch.enable_grad():
                    latent_model_input.requires_grad_(False)
                    noise_pred = self.pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep_tensor,
                        encoder_hidden_states=prompt_embeds_input,
                        pooled_projections=pooled_prompt_embeds_input,
                        return_dict=False,
                    )[0]
                
                # Perform guidance
                if guidance_scale > 1.0:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # Compute the previous noisy sample x_t -> x_t-1
                latents = self.pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
                # Capture latents at specified intervals (move to CPU immediately)
                if i % capture_every_n_steps == 0:
                    self.latent_cache[f"latents_step_{i}"] = latents.detach().cpu()
                    # Clear GPU cache periodically
                    if i > 0 and i % 10 == 0:
                        torch.cuda.empty_cache()
        
        # Decode latents to image
        with torch.no_grad():
            latents = latents / self.pipe.vae.config.scaling_factor
            image = self.pipe.vae.decode(latents, return_dict=False)[0]
            image = self.pipe.image_processor.postprocess(image.detach(), output_type="pil")[0]
        
        # Save artifacts both remotely and locally
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Remote artifacts directory
        artifact_dir = Path(f"/artifacts/{timestamp}")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Save image remotely
        image.save(artifact_dir / "output.png")
        
        # Save image data for local download
        import io
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        image_bytes = img_buffer.getvalue()
        
        # Compute CLIP scores
        clip_scores = self.compute_clip_scores(image, prompt_1, prompt_2)
        
        # Generate comparison images without prompt switching for LPIPS
        print("Generating reference images for LPIPS comparison...")
        with torch.no_grad():
            # Generate with only prompt 1
            reference_1 = self.pipe(
                prompt=prompt_1,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device=self.device).manual_seed(seed) if seed else None,
            ).images[0]
            
            # Generate with only prompt 2
            reference_2 = self.pipe(
                prompt=prompt_2,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device=self.device).manual_seed(seed) if seed else None,
            ).images[0]
        
        # Compute LPIPS distances
        lpips_to_ref1 = self.compute_lpips_distance(image, reference_1)
        lpips_to_ref2 = self.compute_lpips_distance(image, reference_2)
        lpips_ref1_to_ref2 = self.compute_lpips_distance(reference_1, reference_2)
        
        # Analyze attention map variance
        attention_analysis = self.analyze_attention_variance()
        
        # Save reference images
        reference_1.save(artifact_dir / "reference_prompt1.png")
        reference_2.save(artifact_dir / "reference_prompt2.png")
        
        # Save latents and attention maps
        torch.save(self.latent_cache, artifact_dir / "latents.pt")
        torch.save(self.attention_maps, artifact_dir / "attention_maps.pt")
        
        # Save metadata
        metadata = {
            "prompt_1": prompt_1,
            "prompt_2": prompt_2,
            "switch_step": switch_step,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "scheduler": scheduler,
            "timestamp": timestamp,
            "clip_scores": clip_scores,
            "lpips_distances": {
                "to_reference_1": lpips_to_ref1,
                "to_reference_2": lpips_to_ref2,
                "ref1_to_ref2": lpips_ref1_to_ref2,
            },
            "attention_analysis": attention_analysis,
        }
        
        import json
        with open(artifact_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        self.cleanup_hooks()  # Clean up hooks after generation
        
        return {
            "image_path": str(artifact_dir / "output.png"),
            "image_bytes": image_bytes,
            "artifact_dir": str(artifact_dir),
            "metadata": metadata,
            "num_latents_captured": len(self.latent_cache),
            "num_attention_maps": len(self.attention_maps),
        }
    
    @modal.method()
    def run_basic_spectral_analysis(self, artifact_dir: str) -> Dict:
        """Run basic spectral analysis on artifacts."""
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.fft import fft2, fftfreq
        from pathlib import Path
        
        artifact_path = Path(artifact_dir)
        latents = torch.load(artifact_path / "latents.pt")
        
        # Load metadata
        import json
        with open(artifact_path / "metadata.json") as f:
            metadata = json.load(f)
        
        # Basic spectral evolution analysis
        evolution = {"steps": [], "low_freq": [], "mid_freq": [], "high_freq": [], "total_power": []}
        
        for key in sorted(latents.keys()):
            if "latents_step_" in key:
                step = int(key.split("_")[-1])
                latent = latents[key].numpy()[0, 0]  # First channel only
                
                # 2D FFT
                fft_2d = fft2(latent)
                psd = np.abs(fft_2d) ** 2
                
                # Create frequency grid
                freq_y = fftfreq(latent.shape[0])
                freq_x = fftfreq(latent.shape[1])
                freq_r = np.sqrt(freq_y[:, np.newaxis]**2 + freq_x[np.newaxis, :]**2)
                
                # Frequency bands
                total_power = np.sum(psd)
                low_mask = freq_r < 0.1
                mid_mask = (freq_r >= 0.1) & (freq_r < 0.3)
                high_mask = freq_r >= 0.3
                
                evolution["steps"].append(step)
                evolution["total_power"].append(float(total_power))
                evolution["low_freq"].append(float(np.sum(psd[low_mask]) / total_power))
                evolution["mid_freq"].append(float(np.sum(psd[mid_mask]) / total_power))
                evolution["high_freq"].append(float(np.sum(psd[high_mask]) / total_power))
        
        # Simple visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        steps = evolution["steps"]
        switch_step = metadata.get("switch_step", 25)
        
        # Frequency bands
        ax1.plot(steps, evolution["low_freq"], label="Low freq", marker='o')
        ax1.plot(steps, evolution["mid_freq"], label="Mid freq", marker='s')
        ax1.plot(steps, evolution["high_freq"], label="High freq", marker='^')
        ax1.axvline(x=switch_step, color='red', linestyle='--', label='Switch')
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Relative Power")
        ax1.set_title("Frequency Band Evolution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Total power
        ax2.semilogy(steps, evolution["total_power"], 'k-', marker='o')
        ax2.axvline(x=switch_step, color='red', linestyle='--')
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Total Power (log)")
        ax2.set_title("Total Spectral Power")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = artifact_path / "basic_spectral_analysis.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        return {
            "evolution": evolution,
            "plot_path": str(plot_path),
            "analysis_summary": {
                "pre_switch_high_freq": float(np.mean([v for s, v in zip(steps, evolution["high_freq"]) if s < switch_step])) if steps else 0,
                "post_switch_high_freq": float(np.mean([v for s, v in zip(steps, evolution["high_freq"]) if s >= switch_step])) if steps else 0,
            }
        }

    @modal.method()
    def analyze_prompt_switch_artifacts(
        self, artifact_dir: str, analysis_type: str = "latent_diff"
    ) -> Dict:
        """Analyze artifacts from prompt switching experiment."""
        import torch
        import matplotlib.pyplot as plt
        import numpy as np
        from pathlib import Path
        
        artifact_path = Path(artifact_dir)
        latents = torch.load(artifact_path / "latents.pt")
        
        if analysis_type == "latent_diff":
            # Analyze differences in latents before/after switch
            results = {}
            
            for key in sorted(latents.keys()):
                if "latents_step_" in key:
                    step = int(key.split("_")[-1])
                    latent = latents[key]
                    
                    # Compute statistics
                    results[f"step_{step}"] = {
                        "mean": float(latent.mean()),
                        "std": float(latent.std()),
                        "norm": float(torch.norm(latent)),
                    }
            
            # Plot latent statistics over time
            steps = sorted([int(k.split("_")[1]) for k in results.keys()])
            means = [results[f"step_{s}"]["mean"] for s in steps]
            stds = [results[f"step_{s}"]["std"] for s in steps]
            norms = [results[f"step_{s}"]["norm"] for s in steps]
            
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
            
            ax1.plot(steps, means, marker='o')
            ax1.set_xlabel("Step")
            ax1.set_ylabel("Mean")
            ax1.set_title("Latent Mean over Steps")
            ax1.grid(True)
            
            ax2.plot(steps, stds, marker='o', color='orange')
            ax2.set_xlabel("Step")
            ax2.set_ylabel("Std Dev")
            ax2.set_title("Latent Std Dev over Steps")
            ax2.grid(True)
            
            ax3.plot(steps, norms, marker='o', color='green')
            ax3.set_xlabel("Step")
            ax3.set_ylabel("L2 Norm")
            ax3.set_title("Latent L2 Norm over Steps")
            ax3.grid(True)
            
            plt.tight_layout()
            analysis_path = artifact_path / "latent_analysis.png"
            plt.savefig(analysis_path)
            plt.close()
            
            return {
                "analysis_type": analysis_type,
                "results": results,
                "plot_path": str(analysis_path),
            }
        
        elif analysis_type == "block_activations":
            # Analyze per-block activations
            block_stats = {}
            
            for key in latents.keys():
                if "block_" in key and "_step_" in key:
                    parts = key.split("_")
                    block_idx = int(parts[1])
                    step = int(parts[-1])
                    
                    if block_idx not in block_stats:
                        block_stats[block_idx] = {}
                    
                    latent = latents[key]
                    block_stats[block_idx][step] = {
                        "mean": float(latent.mean()),
                        "std": float(latent.std()),
                        "norm": float(torch.norm(latent)),
                    }
            
            return {
                "analysis_type": analysis_type,
                "block_statistics": block_stats,
            }
        
        return {"error": f"Unknown analysis type: {analysis_type}"}
    
    @modal.method()
    def create_comprehensive_analysis_plot(self, artifact_dir: str) -> Dict:
        """Create comprehensive visualization including CLIP, LPIPS, and attention metrics."""
        import matplotlib.pyplot as plt
        import numpy as np
        import json
        from pathlib import Path
        
        artifact_path = Path(artifact_dir)
        
        # Load metadata
        with open(artifact_path / "metadata.json") as f:
            metadata = json.load(f)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. CLIP Scores Visualization
        ax1 = plt.subplot(3, 3, 1)
        clip_scores = metadata.get("clip_scores", {})
        prompts = ['Prompt 1', 'Prompt 2']
        scores = [clip_scores.get("clip_score_prompt_1", 0), clip_scores.get("clip_score_prompt_2", 0)]
        colors = ['blue', 'red']
        bars = ax1.bar(prompts, scores, color=colors, alpha=0.7)
        ax1.set_ylabel('CLIP Score')
        ax1.set_title('CLIP Similarity Scores')
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Add delta annotation
        delta = clip_scores.get("clip_score_delta", 0)
        ax1.text(0.5, 0.9, f'Δ = {delta:+.3f}', transform=ax1.transAxes,
                ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"))
        
        # 2. LPIPS Distances Visualization
        ax2 = plt.subplot(3, 3, 2)
        lpips_data = metadata.get("lpips_distances", {})
        lpips_labels = ['To Ref 1', 'To Ref 2', 'Ref 1 vs 2']
        lpips_values = [
            lpips_data.get("to_reference_1", 0),
            lpips_data.get("to_reference_2", 0),
            lpips_data.get("ref1_to_ref2", 0)
        ]
        bars = ax2.bar(lpips_labels, lpips_values, color=['green', 'orange', 'purple'], alpha=0.7)
        ax2.set_ylabel('LPIPS Distance')
        ax2.set_title('Perceptual Distances (Lower = More Similar)')
        
        # Add value labels
        for bar, val in zip(bars, lpips_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')
        
        # 3. Attention Map Variance Statistics
        ax3 = plt.subplot(3, 3, 3)
        attn_analysis = metadata.get("attention_analysis", {})
        if "error" not in attn_analysis:
            stats = ['Mean', 'Std', 'Max', 'Min']
            values = [
                attn_analysis.get("mean_variance", 0),
                attn_analysis.get("std_variance", 0),
                attn_analysis.get("max_variance", 0),
                attn_analysis.get("min_variance", 0)
            ]
            ax3.bar(stats, values, color='teal', alpha=0.7)
            ax3.set_ylabel('Variance')
            ax3.set_title('Attention Map Variance Statistics')
        else:
            ax3.text(0.5, 0.5, 'No attention data', transform=ax3.transAxes,
                    ha='center', va='center', fontsize=14)
            ax3.set_title('Attention Map Variance')
        
        # 4. Combined Metrics Summary
        ax4 = plt.subplot(3, 3, 4)
        ax4.axis('off')
        
        # Create summary text
        summary_text = f"""Scheduler: {metadata.get('scheduler', 'Unknown')}
Switch Step: {metadata.get('switch_step', 'N/A')}

CLIP Analysis:
  Prompt 1 similarity: {clip_scores.get('clip_score_prompt_1', 0):.3f}
  Prompt 2 similarity: {clip_scores.get('clip_score_prompt_2', 0):.3f}
  Delta (P2-P1): {clip_scores.get('clip_score_delta', 0):+.3f}
  
LPIPS Analysis:
  Distance to Ref 1: {lpips_data.get('to_reference_1', 0):.3f}
  Distance to Ref 2: {lpips_data.get('to_reference_2', 0):.3f}
  Baseline distance: {lpips_data.get('ref1_to_ref2', 0):.3f}
  
Interpretation:
  {'Closer to Prompt 2' if delta > 0 else 'Closer to Prompt 1'}
  Perceptual blend: {(1 - min(lpips_values[:2]) / max(lpips_values[2], 0.001)) * 100:.1f}%"""
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                verticalalignment='top', fontfamily='monospace', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        
        # 5. Attention Variance Heatmap (if available)
        ax5 = plt.subplot(3, 3, 5)
        attn_variances = attn_analysis.get("attention_variances", {})
        if attn_variances and len(attn_variances) > 0:
            # Extract step numbers and create matrix
            steps = sorted(set(int(k.split('_')[-1]) for k in attn_variances.keys() if '_step_' in k))
            blocks = sorted(set(int(k.split('_')[1]) for k in attn_variances.keys() if 'block_' in k))
            
            if steps and blocks:
                variance_matrix = np.zeros((len(blocks), len(steps)))
                for key, var in attn_variances.items():
                    if 'block_' in key and '_step_' in key:
                        parts = key.split('_')
                        block_idx = blocks.index(int(parts[1]))
                        step_idx = steps.index(int(parts[-1]))
                        variance_matrix[block_idx, step_idx] = var
                
                im = ax5.imshow(variance_matrix, aspect='auto', cmap='hot')
                ax5.set_xlabel('Step')
                ax5.set_ylabel('Block')
                ax5.set_title('Attention Variance Heatmap')
                plt.colorbar(im, ax=ax5)
            else:
                ax5.text(0.5, 0.5, 'Insufficient data', transform=ax5.transAxes,
                        ha='center', va='center')
        else:
            ax5.text(0.5, 0.5, 'No variance data', transform=ax5.transAxes,
                    ha='center', va='center')
        
        # 6. Metric Evolution Plot
        ax6 = plt.subplot(3, 1, 3)
        
        # This would require multiple runs to show evolution
        # For now, show a comparison of key metrics
        metrics = ['CLIP Δ', 'LPIPS to R1', 'LPIPS to R2', 'Attn Var (mean)']
        values = [
            abs(clip_scores.get('clip_score_delta', 0)),
            lpips_data.get('to_reference_1', 0),
            lpips_data.get('to_reference_2', 0),
            attn_analysis.get('mean_variance', 0) * 10  # Scale for visibility
        ]
        
        x = np.arange(len(metrics))
        bars = ax6.bar(x, values, color=['blue', 'green', 'orange', 'teal'], alpha=0.7)
        ax6.set_xticks(x)
        ax6.set_xticklabels(metrics)
        ax6.set_ylabel('Normalized Value')
        ax6.set_title('Key Metrics Overview')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = artifact_path / "comprehensive_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            "plot_path": str(plot_path),
            "metrics_summary": {
                "clip_delta": clip_scores.get("clip_score_delta", 0),
                "lpips_to_ref1": lpips_data.get("to_reference_1", 0),
                "lpips_to_ref2": lpips_data.get("to_reference_2", 0),
                "attention_mean_variance": attn_analysis.get("mean_variance", 0),
                "perceptual_blend_ratio": (1 - min(lpips_values[:2]) / max(lpips_values[2], 0.001)) if lpips_values else 0,
            }
        }

    @modal.method()
    def create_switch_comparison(self, results_list: List[Dict]) -> Dict:
        """Create visualization comparing different switch steps."""
        import matplotlib.pyplot as plt
        import numpy as np
        from pathlib import Path
        
        if not results_list:
            return {"error": "No results provided"}
        
        # Extract data for visualization
        switch_steps = []
        hf_changes = []
        images_data = []
        
        for result in results_list:
            if 'switch_step' in result and 'hf_change' in result:
                switch_steps.append(result['switch_step'])
                hf_changes.append(result['hf_change'])
                if 'image_bytes' in result:
                    images_data.append(result['image_bytes'])
        
        if not switch_steps:
            return {"error": "No valid switch step data found"}
        
        # Create comparison visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot high frequency changes
        ax1.bar(switch_steps, hf_changes, alpha=0.7, color=['red' if abs(x) > 0.1 else 'orange' if abs(x) > 0.05 else 'green' for x in hf_changes])
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Strong artifact threshold')
        ax1.axhline(y=-0.1, color='red', linestyle='--', alpha=0.5)
        ax1.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Moderate artifact threshold')
        ax1.axhline(y=-0.05, color='orange', linestyle='--', alpha=0.5)
        
        ax1.set_xlabel('Switch Step')
        ax1.set_ylabel('High Frequency Change')
        ax1.set_title('Prompt Switch Artifact Analysis by Step')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(hf_changes):
            ax1.text(switch_steps[i], v + (0.01 if v >= 0 else -0.02), f'{v:.3f}', 
                    ha='center', va='bottom' if v >= 0 else 'top', fontsize=10)
        
        # Analysis summary
        strongest_artifact = max(hf_changes, key=abs)
        strongest_step = switch_steps[hf_changes.index(strongest_artifact)]
        
        ax2.text(0.05, 0.9, f"Analysis Summary:", fontsize=14, fontweight='bold', transform=ax2.transAxes)
        ax2.text(0.05, 0.8, f"• Total experiments: {len(switch_steps)}", fontsize=12, transform=ax2.transAxes)
        ax2.text(0.05, 0.7, f"• Strongest artifact: {strongest_artifact:.4f} at step {strongest_step}", fontsize=12, transform=ax2.transAxes)
        ax2.text(0.05, 0.6, f"• Strong artifacts (|change| > 0.1): {sum(1 for x in hf_changes if abs(x) > 0.1)}", fontsize=12, transform=ax2.transAxes)
        ax2.text(0.05, 0.5, f"• Moderate artifacts (|change| > 0.05): {sum(1 for x in hf_changes if abs(x) > 0.05)}", fontsize=12, transform=ax2.transAxes)
        ax2.text(0.05, 0.4, f"• Clean generations (|change| ≤ 0.05): {sum(1 for x in hf_changes if abs(x) <= 0.05)}", fontsize=12, transform=ax2.transAxes)
        
        # Interpretation
        ax2.text(0.05, 0.25, "Interpretation:", fontsize=14, fontweight='bold', transform=ax2.transAxes)
        if strongest_step <= 15:
            interpretation = "Early switching creates strongest artifacts (high-level structure conflicts)"
        elif strongest_step >= 35:
            interpretation = "Late switching creates strongest artifacts (fine detail conflicts)"
        else:
            interpretation = "Mid-generation switching creates strongest artifacts (semantic transition)"
        ax2.text(0.05, 0.15, f"• {interpretation}", fontsize=12, transform=ax2.transAxes)
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = results_list[0].get('timestamp', 'unknown')
        plot_path = f"/artifacts/switch_comparison_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            "plot_path": plot_path,
            "analysis": {
                "total_experiments": len(switch_steps),
                "strongest_artifact": float(strongest_artifact),
                "strongest_step": int(strongest_step),
                "strong_artifacts": int(sum(1 for x in hf_changes if abs(x) > 0.1)),
                "moderate_artifacts": int(sum(1 for x in hf_changes if abs(x) > 0.05)),
                "clean_generations": int(sum(1 for x in hf_changes if abs(x) <= 0.05)),
                "interpretation": interpretation,
            }
        }


@app.local_entrypoint()
def main(
    prompt_1: str = "A photo of a cat sitting on a mat",
    prompt_2: str = "A photo of a dog playing in the park",
    switch_step: int = 25,
    seed: int = 42,
    scheduler: str = "FlowMatchEulerDiscreteScheduler",
):
    """Run SD3 prompt switching experiment."""
    sd3 = SD3MechInterp()
    
    print(f"Generating with prompt switch at step {switch_step}...")
    print(f"Prompt 1: {prompt_1}")
    print(f"Prompt 2: {prompt_2}")
    print(f"Scheduler: {scheduler}")
    
    result = sd3.generate_with_prompt_switch.remote(
        prompt_1=prompt_1,
        prompt_2=prompt_2,
        switch_step=switch_step,
        seed=seed,
        scheduler=scheduler,
    )
    
    print(f"\nResults saved to: {result['artifact_dir']}")
    print(f"Captured {result['num_latents_captured']} latent states")
    print(f"Captured {result['num_attention_maps']} attention maps")
    
    # Run analysis
    print("\nAnalyzing artifacts...")
    analysis = sd3.analyze_prompt_switch_artifacts.remote(
        result['artifact_dir'], 
        analysis_type="latent_diff"
    )
    print(f"Analysis saved to: {analysis.get('plot_path', 'N/A')}")
    
    # Run basic spectral analysis
    print("\nRunning basic spectral analysis...")
    spectral_results = sd3.run_basic_spectral_analysis.remote(result['artifact_dir'])
    print(f"Spectral plot saved to: {spectral_results['plot_path']}")
    
    # Print key spectral findings
    summary = spectral_results['analysis_summary']
    print(f"Pre-switch high freq: {summary['pre_switch_high_freq']:.4f}")
    print(f"Post-switch high freq: {summary['post_switch_high_freq']:.4f}")
    print(f"High freq change: {summary['post_switch_high_freq'] - summary['pre_switch_high_freq']:+.4f}")
    
    # Run comprehensive analysis
    print("\nCreating comprehensive analysis visualization...")
    comp_analysis = sd3.create_comprehensive_analysis_plot.remote(result['artifact_dir'])
    print(f"Comprehensive analysis saved to: {comp_analysis['plot_path']}")
    
    # Print key metrics
    metrics = comp_analysis['metrics_summary']
    print("\nKey Metrics Summary:")
    print(f"  CLIP delta: {metrics['clip_delta']:+.3f}")
    print(f"  LPIPS to reference 1: {metrics['lpips_to_ref1']:.3f}")
    print(f"  LPIPS to reference 2: {metrics['lpips_to_ref2']:.3f}")
    print(f"  Attention variance (mean): {metrics['attention_mean_variance']:.4f}")
    print(f"  Perceptual blend ratio: {metrics['perceptual_blend_ratio']:.1f}%")
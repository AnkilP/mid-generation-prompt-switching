"""Experiments for understanding SD3 prompt switching artifacts."""
import modal
import itertools
from pathlib import Path
from typing import List, Dict, Tuple
import json
import numpy as np
from datetime import datetime


def run_systematic_experiments(
    prompt_pairs: List[Tuple[str, str]],
    switch_steps: List[int] = [10, 20, 30, 40],
    seeds: List[int] = [42, 123, 456],
    num_inference_steps: int = 50,
) -> Dict[str, List[str]]:
    """Run systematic experiments varying prompt pairs, switch steps, and seeds."""
    from src.modal.sd3_modal import SD3MechInterp
    
    sd3 = SD3MechInterp()
    results = {
        "successful": [],
        "failed": [],
        "experiment_config": {
            "num_prompt_pairs": len(prompt_pairs),
            "switch_steps": switch_steps,
            "seeds": seeds,
            "total_experiments": len(prompt_pairs) * len(switch_steps) * len(seeds),
        }
    }
    
    for (prompt_1, prompt_2), switch_step, seed in itertools.product(
        prompt_pairs, switch_steps, seeds
    ):
        try:
            print(f"\nRunning: {prompt_1[:30]}... → {prompt_2[:30]}... @ step {switch_step}, seed {seed}")
            
            result = sd3.generate_with_prompt_switch.remote(
                prompt_1=prompt_1,
                prompt_2=prompt_2,
                switch_step=switch_step,
                seed=seed,
                num_inference_steps=num_inference_steps,
                capture_every_n_steps=5,
            )
            
            results["successful"].append({
                "artifact_dir": result["artifact_dir"],
                "prompt_1": prompt_1,
                "prompt_2": prompt_2,
                "switch_step": switch_step,
                "seed": seed,
            })
            
        except Exception as e:
            print(f"Failed: {e}")
            results["failed"].append({
                "error": str(e),
                "prompt_1": prompt_1,
                "prompt_2": prompt_2,
                "switch_step": switch_step,
                "seed": seed,
            })
    
    # Save experiment manifest
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = Path(f"data/experiments/manifest_{timestamp}.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nExperiment manifest saved to: {manifest_path}")
    return results


def analyze_artifact_patterns(manifest_path: str) -> Dict:
    """Analyze patterns in artifacts across experiments."""
    from src.analysis.artifact_detector import ArtifactDetector, compare_experiments
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    analysis_results = {
        "by_switch_step": {},
        "by_prompt_similarity": {},
        "consistency_metrics": {},
    }
    
    # Group experiments by switch step
    for exp in manifest["successful"]:
        switch_step = exp["switch_step"]
        if switch_step not in analysis_results["by_switch_step"]:
            analysis_results["by_switch_step"][switch_step] = []
        analysis_results["by_switch_step"][switch_step].append(exp["artifact_dir"])
    
    # Analyze consistency within each switch step group
    for switch_step, dirs in analysis_results["by_switch_step"].items():
        if len(dirs) > 1:
            comparisons = compare_experiments(dirs)
            
            # Compute consistency metrics
            discontinuity_std = np.std(list(comparisons["discontinuity_counts"].values()))
            avg_discontinuity_std = np.std(list(comparisons["avg_switch_discontinuity"].values()))
            
            analysis_results["consistency_metrics"][switch_step] = {
                "discontinuity_count_std": float(discontinuity_std),
                "avg_discontinuity_std": float(avg_discontinuity_std),
                "num_experiments": len(dirs),
            }
    
    return analysis_results


def generate_prompt_pairs_for_similarity_study() -> List[Tuple[str, str]]:
    """Generate prompt pairs with varying semantic similarity."""
    base_prompts = [
        "A photo of a cat",
        "A painting of a landscape",
        "A portrait of a person",
        "Abstract geometric shapes",
        "A futuristic city",
    ]
    
    modifiers = [
        ("sitting on a mat", "playing with yarn"),  # Similar context
        ("in bright daylight", "under moonlight"),  # Opposite lighting
        ("in realistic style", "in cartoon style"),  # Different styles
        ("with warm colors", "with cool colors"),    # Color shift
        ("indoors", "outdoors"),                    # Location change
    ]
    
    prompt_pairs = []
    
    # Similar pairs (small changes)
    for base in base_prompts[:3]:
        for mod1, mod2 in modifiers[:2]:
            prompt_pairs.append((f"{base} {mod1}", f"{base} {mod2}"))
    
    # Different pairs (larger changes)
    for i, base1 in enumerate(base_prompts):
        for j, base2 in enumerate(base_prompts):
            if i < j:  # Avoid duplicates
                prompt_pairs.append((base1, base2))
    
    # Style/attribute transfers
    for base in base_prompts[:2]:
        for mod1, mod2 in modifiers[2:]:
            prompt_pairs.append((f"{base} {mod1}", f"{base} {mod2}"))
    
    return prompt_pairs


def identify_robust_artifacts(
    manifest_path: str,
    min_consistency_threshold: float = 0.8
) -> Dict:
    """Identify artifacts that appear consistently across experiments."""
    from src.analysis.artifact_detector import ArtifactDetector
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Analyze each experiment
    all_artifacts = []
    
    for exp in manifest["successful"]:
        detector = ArtifactDetector(exp["artifact_dir"])
        
        # Get all detected artifacts
        discontinuities = detector.detect_discontinuities()
        block_analysis = detector.detect_block_artifacts()
        
        # Extract artifact features
        artifact_features = {
            "switch_step": exp["switch_step"],
            "has_mean_jump": len(discontinuities["mean_jumps"]) > 0,
            "has_norm_jump": len(discontinuities["norm_jumps"]) > 0,
            "has_cosine_jump": len(discontinuities["cosine_jumps"]) > 0,
            "num_affected_blocks": sum(
                1 for data in block_analysis.values() 
                if "metrics" in data and data["metrics"]["switch_discontinuity"] > 0.1
            ),
            "max_block_discontinuity": max(
                (data["metrics"]["switch_discontinuity"] 
                 for data in block_analysis.values() if "metrics" in data),
                default=0
            ),
        }
        
        all_artifacts.append(artifact_features)
    
    # Find consistent patterns
    consistent_patterns = {}
    
    # Group by switch step
    by_switch_step = {}
    for artifact in all_artifacts:
        step = artifact["switch_step"]
        if step not in by_switch_step:
            by_switch_step[step] = []
        by_switch_step[step].append(artifact)
    
    # Check consistency for each artifact type
    for step, artifacts in by_switch_step.items():
        if len(artifacts) < 3:
            continue
        
        total = len(artifacts)
        consistent_patterns[step] = {
            "mean_jump_rate": sum(a["has_mean_jump"] for a in artifacts) / total,
            "norm_jump_rate": sum(a["has_norm_jump"] for a in artifacts) / total,
            "cosine_jump_rate": sum(a["has_cosine_jump"] for a in artifacts) / total,
            "avg_affected_blocks": np.mean([a["num_affected_blocks"] for a in artifacts]),
            "std_affected_blocks": np.std([a["num_affected_blocks"] for a in artifacts]),
            "total_experiments": total,
        }
    
    # Identify most consistent artifacts
    robust_artifacts = {
        "highly_consistent": [],
        "moderately_consistent": [],
        "inconsistent": [],
    }
    
    for step, patterns in consistent_patterns.items():
        for artifact_type in ["mean_jump_rate", "norm_jump_rate", "cosine_jump_rate"]:
            rate = patterns[artifact_type]
            if rate >= min_consistency_threshold:
                robust_artifacts["highly_consistent"].append({
                    "type": artifact_type.replace("_rate", ""),
                    "switch_step": step,
                    "consistency": rate,
                    "num_experiments": patterns["total_experiments"],
                })
            elif rate >= 0.5:
                robust_artifacts["moderately_consistent"].append({
                    "type": artifact_type.replace("_rate", ""),
                    "switch_step": step,
                    "consistency": rate,
                    "num_experiments": patterns["total_experiments"],
                })
    
    return {
        "consistent_patterns": consistent_patterns,
        "robust_artifacts": robust_artifacts,
        "summary": {
            "total_experiments": len(all_artifacts),
            "highly_consistent_count": len(robust_artifacts["highly_consistent"]),
            "moderately_consistent_count": len(robust_artifacts["moderately_consistent"]),
        }
    }


def run_targeted_artifact_study(
    artifact_type: str = "cosine_jump",
    num_experiments: int = 10
) -> Dict:
    """Run targeted experiments to better understand specific artifact types."""
    # Design experiments that should maximize the chosen artifact
    if artifact_type == "cosine_jump":
        # Use very different prompts to maximize cosine distance
        prompt_pairs = [
            ("A photo of a serene mountain lake", "Abstract neon cyberpunk patterns"),
            ("Classical oil painting of flowers", "Futuristic spaceship in deep space"),
            ("Cute cartoon animal character", "Dark gothic cathedral interior"),
            ("Bright sunny beach scene", "Underground cave with glowing crystals"),
            ("Modern minimalist architecture", "Dense jungle with exotic wildlife"),
        ]
        switch_steps = [20, 25, 30]  # Mid-generation switches
        
    elif artifact_type == "frequency_artifact":
        # Use prompts that differ in texture/frequency content
        prompt_pairs = [
            ("Smooth gradient background", "Detailed fractal patterns"),
            ("Soft blurred portrait", "Sharp geometric tessellation"),
            ("Calm water surface", "Chaotic lightning storm"),
            ("Simple color blocks", "Intricate mandala design"),
            ("Foggy landscape", "High contrast zebra stripes"),
        ]
        switch_steps = [15, 25, 35]
        
    else:
        raise ValueError(f"Unknown artifact type: {artifact_type}")
    
    # Run experiments with multiple seeds for robustness
    seeds = list(range(num_experiments))
    
    results = run_systematic_experiments(
        prompt_pairs=prompt_pairs[:3],  # Use subset for targeted study
        switch_steps=switch_steps,
        seeds=seeds,
    )
    
    return results


if __name__ == "__main__":
    # Example: Run a systematic study
    prompt_pairs = generate_prompt_pairs_for_similarity_study()[:5]  # Start with 5 pairs
    
    print("Running systematic prompt switching experiments...")
    results = run_systematic_experiments(
        prompt_pairs=prompt_pairs,
        switch_steps=[15, 25, 35],
        seeds=[42, 123, 456],
    )
    
    print(f"\nSuccessful experiments: {len(results['successful'])}")
    print(f"Failed experiments: {len(results['failed'])}")
    
    # Analyze patterns if we have results
    if results['successful']:
        manifest_path = sorted(Path("data/experiments").glob("manifest_*.json"))[-1]
        
        print("\nAnalyzing artifact patterns...")
        patterns = identify_robust_artifacts(str(manifest_path))
        
        print("\nRobust artifacts found:")
        for artifact in patterns["robust_artifacts"]["highly_consistent"]:
            print(f"- {artifact['type']} at step {artifact['switch_step']}: "
                  f"{artifact['consistency']:.1%} consistency")
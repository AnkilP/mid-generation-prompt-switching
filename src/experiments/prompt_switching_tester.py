"""Unified prompt switching experiment tester."""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import itertools


@dataclass
class PromptSwitchingConfig:
    """Configuration for prompt switching experiments."""
    # Experiment identification
    name: str = "prompt_switching_experiment"
    description: str = "Prompt switching experiment"
    
    # Prompts and parameters
    prompt_pairs: List[Tuple[str, str]] = field(default_factory=list)
    switch_steps: List[int] = field(default_factory=lambda: [20, 30])
    seeds: List[int] = field(default_factory=lambda: [42])
    
    # Generation parameters
    num_inference_steps: int = 50
    guidance_scale: float = 7.0
    capture_every_n_steps: int = 5
    
    # Output and analysis
    output_dir: str = "data/experiments"
    save_artifacts: bool = True
    run_analysis: bool = True
    analysis_types: List[str] = field(default_factory=lambda: ["latent_diff", "spectral"])
    
    # Visualization settings
    create_visualizations: bool = True
    visualization_types: List[str] = field(default_factory=lambda: ["experiment", "comparison", "summary", "grid"])
    
    # Modal compute settings
    gpu_type: str = "a100"
    timeout: int = 1200
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PromptSwitchingConfig":
        """Load configuration from YAML file."""
        import yaml  # Import only when needed
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "PromptSwitchingConfig":
        """Create configuration from dictionary."""
        return cls(**data)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def save_yaml(self, path: Union[str, Path]):
        """Save configuration to YAML file."""
        import yaml  # Import only when needed
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


class PromptSwitchingTester:
    """
    Unified tester for prompt switching experiments.
    
    This class handles the complete workflow:
    1. Configuration management
    2. Running experiments on Modal
    3. Collecting and organizing results
    4. Analyzing artifacts and patterns
    5. Generating reports
    
    Example:
        # Quick test with default config
        tester = PromptSwitchingTester()
        tester.add_prompt_pair("A cat", "A dog")
        tester.run()
        
        # From configuration
        config = PromptSwitchingConfig(name="my_experiment")
        tester = PromptSwitchingTester(config)
        tester.run()
        
        # From YAML
        tester = PromptSwitchingTester.from_yaml("config.yaml")
        tester.run()
    """
    
    # Preset experiment configurations
    PRESETS = {
        "quick": {
            "name": "quick_test",
            "description": "Quick test with minimal configuration",
            "prompt_pairs": [
                ("A photo of a cat", "A photo of a dog"),
                ("Day scene", "Night scene"),
                ("Abstract art", "Realistic portrait"),
            ],
            "switch_steps": [20, 30],
            "seeds": [42],
        },
        
        "contrast": {
            "name": "high_contrast",
            "description": "High contrast prompts for maximum artifacts",
            "prompt_pairs": [
                ("Peaceful zen garden", "Chaotic battlefield"),
                ("Minimalist white room", "Maximalist colorful bazaar"),
                ("Smooth glass surface", "Rough tree bark"),
                ("Silent library", "Rock concert"),
                ("Empty desert", "Crowded city"),
            ],
            "switch_steps": [10, 20, 30, 40],
            "seeds": [42, 123],
        },
        
        "style": {
            "name": "style_transfer",
            "description": "Same content in different styles",
            "prompt_pairs": [
                ("Mountain in photorealistic style", "Mountain in impressionist style"),
                ("Cat in realistic style", "Cat in cartoon style"),
                ("Portrait in oil painting", "Portrait in watercolor"),
                ("City in architectural drawing", "City in abstract style"),
                ("Flower in botanical illustration", "Flower in pop art style"),
            ],
            "switch_steps": [15, 25, 35],
            "seeds": [42, 123, 456],
        },
        
        "frequency": {
            "name": "frequency_artifacts",
            "description": "Test frequency and texture artifacts",
            "prompt_pairs": [
                ("Smooth gradient", "Sharp checkerboard pattern"),
                ("Soft bokeh blur", "Crisp geometric lines"),
                ("Low frequency waves", "High frequency noise"),
                ("Simple flat color", "Complex fractal pattern"),
                ("Gaussian blur", "Motion blur"),
            ],
            "switch_steps": [10, 20, 30, 40],
            "seeds": [42, 123],
        },
        
        "semantic": {
            "name": "semantic_distance",
            "description": "Varying semantic distances between prompts",
            "prompt_pairs": [
                # Very similar
                ("Red apple", "Green apple"),
                ("Happy person", "Smiling person"),
                # Somewhat similar  
                ("House cat", "Lion"),
                ("Lake", "Ocean"),
                # Different but related
                ("Summer", "Winter"),
                ("Ancient", "Modern"),
                # Very different
                ("Fire", "Ice"),
                ("Microscopic", "Cosmic"),
            ],
            "switch_steps": [20, 30],
            "seeds": [42, 123, 456],
        },
    }
    
    def __init__(self, config: Optional[Union[PromptSwitchingConfig, Dict, str]] = None):
        """
        Initialize the tester.
        
        Args:
            config: Configuration (PromptSwitchingConfig, dict, YAML path, or None for default)
        """
        if config is None:
            self.config = PromptSwitchingConfig()
        elif isinstance(config, str):
            self.config = PromptSwitchingConfig.from_yaml(config)
        elif isinstance(config, dict):
            self.config = PromptSwitchingConfig.from_dict(config)
        else:
            self.config = config
        
        self.results = {
            "config": self.config.to_dict(),
            "successful": [],
            "failed": [],
            "analysis": {},
            "metadata": {
                "start_time": None,
                "end_time": None,
                "total_experiments": 0,
            }
        }
        
        self._sd3 = None
        self._visualizer = None
    
    @classmethod
    def from_preset(cls, preset_name: str) -> "PromptSwitchingTester":
        """Create tester from a preset configuration."""
        if preset_name not in cls.PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(cls.PRESETS.keys())}")
        
        config = PromptSwitchingConfig(**cls.PRESETS[preset_name])
        return cls(config)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PromptSwitchingTester":
        """Create tester from YAML configuration file."""
        config = PromptSwitchingConfig.from_yaml(path)
        return cls(config)
    
    @property
    def sd3(self):
        """Lazy load SD3 Modal app."""
        if self._sd3 is None:
            from src.modal.sd3_modal import SD3MechInterp
            self._sd3 = SD3MechInterp()
        return self._sd3
    
    @property
    def visualizer(self):
        """Lazy load visualizer."""
        if self._visualizer is None:
            from src.visualization.prompt_switch_visualizer import PromptSwitchVisualizer
            viz_dir = Path(self.config.output_dir) / "visualizations"
            self._visualizer = PromptSwitchVisualizer(str(viz_dir))
        return self._visualizer
    
    def add_prompt_pair(self, prompt_1: str, prompt_2: str):
        """Add a single prompt pair."""
        self.config.prompt_pairs.append((prompt_1, prompt_2))
        return self
    
    def add_prompt_pairs(self, pairs: List[Tuple[str, str]]):
        """Add multiple prompt pairs."""
        self.config.prompt_pairs.extend(pairs)
        return self
    
    def set_switch_steps(self, steps: List[int]):
        """Set switch steps."""
        self.config.switch_steps = steps
        return self
    
    def set_seeds(self, seeds: List[int]):
        """Set random seeds."""
        self.config.seeds = seeds
        return self
    
    def set_output_dir(self, path: str):
        """Set output directory."""
        self.config.output_dir = path
        return self
    
    def _run_single_experiment(self, prompt_1: str, prompt_2: str, switch_step: int, seed: int) -> Dict:
        """Run a single experiment."""
        try:
            print(f"  → {prompt_1[:30]}... → {prompt_2[:30]}... (step {switch_step}, seed {seed})")
            
            result = self.sd3.generate_with_prompt_switch.remote(
                prompt_1=prompt_1,
                prompt_2=prompt_2,
                switch_step=switch_step,
                seed=seed,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                capture_every_n_steps=self.config.capture_every_n_steps,
            )
            
            experiment_result = {
                "prompt_1": prompt_1,
                "prompt_2": prompt_2,
                "switch_step": switch_step,
                "seed": seed,
                "artifact_dir": result["artifact_dir"],
                "image_path": result["image_path"],
                "num_latents_captured": result["num_latents_captured"],
                "num_attention_maps": result["num_attention_maps"],
                "timestamp": datetime.now().isoformat(),
            }
            
            # Run immediate analysis if configured
            if self.config.run_analysis and "spectral" in self.config.analysis_types:
                try:
                    spectral_result = self.sd3.run_basic_spectral_analysis.remote(
                        result["artifact_dir"]
                    )
                    experiment_result["spectral_analysis"] = spectral_result
                except Exception as e:
                    print(f"    ⚠️  Spectral analysis failed: {e}")
            
            self.results["successful"].append(experiment_result)
            print(f"    ✓ Success")
            return experiment_result
            
        except Exception as e:
            error_result = {
                "prompt_1": prompt_1,
                "prompt_2": prompt_2,
                "switch_step": switch_step,
                "seed": seed,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            self.results["failed"].append(error_result)
            print(f"    ✗ Failed: {e}")
            return error_result
    
    def run(self) -> Dict:
        """Run all configured experiments."""
        if not self.config.prompt_pairs:
            raise ValueError("No prompt pairs configured. Use add_prompt_pair() or load a config.")
        
        self.results["metadata"]["start_time"] = datetime.now().isoformat()
        
        total = len(self.config.prompt_pairs) * len(self.config.switch_steps) * len(self.config.seeds)
        self.results["metadata"]["total_experiments"] = total
        
        print(f"\n{'='*60}")
        print(f"Running: {self.config.name}")
        print(f"Description: {self.config.description}")
        print(f"Total experiments: {total}")
        print(f"{'='*60}\n")
        
        experiment_count = 0
        
        for (prompt_1, prompt_2), switch_step, seed in itertools.product(
            self.config.prompt_pairs, self.config.switch_steps, self.config.seeds
        ):
            experiment_count += 1
            print(f"[{experiment_count}/{total}]")
            self._run_single_experiment(prompt_1, prompt_2, switch_step, seed)
        
        self.results["metadata"]["end_time"] = datetime.now().isoformat()
        
        # Save results
        self._save_results()
        
        # Run analysis if configured
        if self.config.run_analysis and self.results["successful"]:
            self.analyze()
        
        # Print summary
        self._print_summary()
        
        # Create visualizations if configured
        if self.config.create_visualizations and self.results["successful"]:
            self.visualize()
        
        return self.results
    
    def analyze(self) -> Dict:
        """Analyze experiment results."""
        if not self.results["successful"]:
            print("No successful experiments to analyze.")
            return {}
        
        print("\n🔍 Analyzing results...")
        
        # Group experiments by switch step
        by_switch_step = {}
        for exp in self.results["successful"]:
            step = exp["switch_step"]
            if step not in by_switch_step:
                by_switch_step[step] = []
            by_switch_step[step].append(exp)
        
        # Analyze consistency by switch step
        switch_step_analysis = {}
        for switch_step, experiments in by_switch_step.items():
            if len(experiments) >= 2 and "spectral_analysis" in experiments[0]:
                # Extract high frequency changes
                high_freq_changes = []
                for exp in experiments:
                    if "spectral_analysis" in exp:
                        summary = exp["spectral_analysis"]["analysis_summary"]
                        change = summary["post_switch_high_freq"] - summary["pre_switch_high_freq"]
                        high_freq_changes.append(change)
                
                if high_freq_changes:
                    import numpy as np  # Import only when needed
                    switch_step_analysis[switch_step] = {
                        "num_experiments": len(experiments),
                        "high_freq_change_mean": float(np.mean(high_freq_changes)),
                        "high_freq_change_std": float(np.std(high_freq_changes)),
                        "consistency_score": float(np.std(high_freq_changes) / (np.abs(np.mean(high_freq_changes)) + 1e-6)),
                    }
        
        # Identify robust artifacts
        robust_artifacts = []
        for switch_step, stats in switch_step_analysis.items():
            if stats["consistency_score"] < 0.3:  # High consistency
                robust_artifacts.append({
                    "switch_step": switch_step,
                    "artifact_type": "high_frequency_change",
                    "mean_change": stats["high_freq_change_mean"],
                    "consistency_score": stats["consistency_score"],
                    "num_experiments": stats["num_experiments"],
                })
        
        self.results["analysis"] = {
            "by_switch_step": switch_step_analysis,
            "robust_artifacts": sorted(robust_artifacts, key=lambda x: x["consistency_score"]),
            "summary": {
                "total_analyzed": len(self.results["successful"]),
                "switch_steps_analyzed": list(switch_step_analysis.keys()),
                "robust_artifact_count": len(robust_artifacts),
            }
        }
        
        # Run additional analyses
        if "latent_diff" in self.config.analysis_types:
            self._analyze_latent_differences()
        
        return self.results["analysis"]
    
    def _analyze_latent_differences(self):
        """Analyze latent space differences around switch points."""
        # This would analyze the saved latent tensors
        # For now, we'll add a placeholder
        self.results["analysis"]["latent_analysis"] = {
            "status": "Would analyze latent discontinuities from saved tensors",
            "metrics": ["mean_jump", "norm_jump", "cosine_similarity"],
        }
    
    def create_report(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        """Create a detailed markdown report."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"report_{self.config.name}_{timestamp}.md"
        
        output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            # Header
            f.write(f"# {self.config.name}\n\n")
            f.write(f"**Description:** {self.config.description}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Configuration summary
            f.write("## Configuration\n\n")
            f.write(f"- **Prompt pairs:** {len(self.config.prompt_pairs)}\n")
            f.write(f"- **Switch steps:** {self.config.switch_steps}\n")
            f.write(f"- **Seeds:** {self.config.seeds}\n")
            f.write(f"- **Total experiments:** {self.results['metadata']['total_experiments']}\n")
            f.write(f"- **Inference steps:** {self.config.num_inference_steps}\n")
            f.write(f"- **Guidance scale:** {self.config.guidance_scale}\n\n")
            
            # Results summary
            f.write("## Results\n\n")
            f.write(f"- ✅ **Successful:** {len(self.results['successful'])}\n")
            f.write(f"- ❌ **Failed:** {len(self.results['failed'])}\n\n")
            
            # Analysis results
            if self.results.get("analysis"):
                f.write("## Analysis\n\n")
                
                analysis = self.results["analysis"]
                if "robust_artifacts" in analysis and analysis["robust_artifacts"]:
                    f.write("### Robust Artifacts\n\n")
                    f.write("Artifacts that appear consistently across experiments:\n\n")
                    
                    for artifact in analysis["robust_artifacts"][:5]:
                        f.write(f"- **Switch step {artifact['switch_step']}**: "
                               f"{artifact['artifact_type']} "
                               f"(consistency: {artifact['consistency_score']:.3f}, "
                               f"n={artifact['num_experiments']})\n")
                
                if "by_switch_step" in analysis:
                    f.write("\n### Switch Step Analysis\n\n")
                    f.write("| Switch Step | Experiments | Mean HF Change | Std Dev | Consistency |\n")
                    f.write("|------------|-------------|----------------|---------|-------------|\n")
                    
                    for step, stats in sorted(analysis["by_switch_step"].items()):
                        f.write(f"| {step} | {stats['num_experiments']} | "
                               f"{stats['high_freq_change_mean']:.4f} | "
                               f"{stats['high_freq_change_std']:.4f} | "
                               f"{stats['consistency_score']:.3f} |\n")
            
            # Detailed results
            if self.results["successful"]:
                f.write("\n## Experiment Details\n\n")
                f.write("First 20 successful experiments:\n\n")
                f.write("| Prompt 1 | Prompt 2 | Switch | Seed | Artifacts |\n")
                f.write("|----------|----------|--------|------|----------|\n")
                
                for exp in self.results["successful"][:20]:
                    p1 = exp["prompt_1"][:25] + "..." if len(exp["prompt_1"]) > 25 else exp["prompt_1"]
                    p2 = exp["prompt_2"][:25] + "..." if len(exp["prompt_2"]) > 25 else exp["prompt_2"]
                    artifacts = exp["artifact_dir"].split("/")[-1]
                    f.write(f"| {p1} | {p2} | {exp['switch_step']} | {exp['seed']} | {artifacts} |\n")
            
            # Failed experiments
            if self.results["failed"]:
                f.write("\n## Failed Experiments\n\n")
                for exp in self.results["failed"][:5]:
                    f.write(f"- **Error:** {exp['error']}\n")
                    f.write(f"  - Prompts: '{exp['prompt_1']}' → '{exp['prompt_2']}'\n")
                    f.write(f"  - Switch: {exp['switch_step']}, Seed: {exp['seed']}\n\n")
        
        print(f"\n📄 Report saved: {output_path}")
        return output_path
    
    def _save_results(self):
        """Save experiment results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = output_dir / f"results_{self.config.name}_{timestamp}.json"
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved: {results_path}")
        return results_path
    
    def _print_summary(self):
        """Print experiment summary."""
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Successful: {len(self.results['successful'])}")
        print(f"❌ Failed: {len(self.results['failed'])}")
        
        if self.results.get("analysis") and "robust_artifacts" in self.results["analysis"]:
            artifacts = self.results["analysis"]["robust_artifacts"]
            if artifacts:
                print(f"\n🎯 Found {len(artifacts)} robust artifacts")
                for artifact in artifacts[:3]:
                    print(f"   - Step {artifact['switch_step']}: {artifact['artifact_type']}")
    
    def visualize(self) -> Dict[str, List[Path]]:
        """Create visualizations for experiment results."""
        print("\n🎨 Creating visualizations...")
        
        viz_paths = {
            "experiment": [],
            "comparison": [],
            "summary": [],
            "grid": []
        }
        
        # Individual experiment visualizations
        if "experiment" in self.config.visualization_types:
            print("  Creating individual experiment visualizations...")
            for i, exp in enumerate(self.results["successful"][:10]):  # Limit to first 10
                try:
                    viz_path = self.visualizer.visualize_experiment(exp)
                    viz_paths["experiment"].append(viz_path)
                    if i == 0:
                        print(f"    Example saved to: {viz_path}")
                except Exception as e:
                    print(f"    Warning: Failed to visualize experiment {i}: {e}")
        
        # Comparison visualization
        if "comparison" in self.config.visualization_types and len(self.results["successful"]) >= 2:
            print("  Creating comparison visualization...")
            try:
                # Compare experiments with same prompts but different switch steps
                comparisons = self._group_for_comparison()
                for prompt_key, experiments in comparisons.items():
                    if len(experiments) >= 2:
                        title = f"Comparison: {prompt_key[0][:30]}... → {prompt_key[1][:30]}..."
                        viz_path = self.visualizer.visualize_comparison(
                            experiments[:4],  # Limit to 4 for space
                            title=title
                        )
                        viz_paths["comparison"].append(viz_path)
                        break  # Just do one comparison for demo
                
                if viz_paths["comparison"]:
                    print(f"    Saved to: {viz_paths['comparison'][0]}")
            except Exception as e:
                print(f"    Warning: Failed to create comparison: {e}")
        
        # Summary visualization
        if "summary" in self.config.visualization_types and self.results.get("analysis"):
            print("  Creating summary visualization...")
            try:
                viz_path = self.visualizer.visualize_artifact_summary(self.results["analysis"])
                viz_paths["summary"].append(viz_path)
                print(f"    Saved to: {viz_path}")
            except Exception as e:
                print(f"    Warning: Failed to create summary: {e}")
        
        # Grid visualization
        if "grid" in self.config.visualization_types:
            print("  Creating experiment grid...")
            try:
                viz_path = self.visualizer.create_experiment_grid(
                    self.results["successful"][:20]  # Limit to 20 images
                )
                viz_paths["grid"].append(viz_path)
                print(f"    Saved to: {viz_path}")
            except Exception as e:
                print(f"    Warning: Failed to create grid: {e}")
        
        # Save visualization paths with results
        self.results["visualizations"] = viz_paths
        
        # Print summary
        total_viz = sum(len(paths) for paths in viz_paths.values())
        print(f"\n  Created {total_viz} visualizations")
        
        return viz_paths
    
    def _group_for_comparison(self) -> Dict[Tuple[str, str], List[Dict]]:
        """Group experiments by prompt pair for comparison."""
        groups = {}
        for exp in self.results["successful"]:
            key = (exp["prompt_1"], exp["prompt_2"])
            if key not in groups:
                groups[key] = []
            groups[key].append(exp)
        
        # Sort experiments within each group by switch step
        for key in groups:
            groups[key].sort(key=lambda x: x["switch_step"])
        
        return groups


def main():
    """CLI interface for the tester."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run prompt switching experiments")
    parser.add_argument("mode", nargs="?", default="quick",
                       help="Preset name, 'custom', or path to config YAML")
    parser.add_argument("--list-presets", action="store_true",
                       help="List available presets")
    parser.add_argument("--no-analysis", action="store_true",
                       help="Skip analysis after experiments")
    parser.add_argument("--output-dir", help="Override output directory")
    
    args = parser.parse_args()
    
    if args.list_presets:
        print("Available presets:")
        for name, preset in PromptSwitchingTester.PRESETS.items():
            print(f"\n{name}:")
            print(f"  {preset['description']}")
            print(f"  Prompt pairs: {len(preset['prompt_pairs'])}")
        return
    
    # Create tester based on mode
    if args.mode in PromptSwitchingTester.PRESETS:
        tester = PromptSwitchingTester.from_preset(args.mode)
    elif args.mode.endswith('.yaml') or args.mode.endswith('.yml'):
        tester = PromptSwitchingTester.from_yaml(args.mode)
    elif args.mode == "custom":
        # Create empty tester for manual configuration
        tester = PromptSwitchingTester()
        print("Created empty tester. Add prompts with add_prompt_pair() before running.")
        return
    else:
        print(f"Unknown mode: {args.mode}")
        print(f"Available presets: {list(PromptSwitchingTester.PRESETS.keys())}")
        print("Or provide a path to a YAML config file")
        return
    
    # Override settings if provided
    if args.output_dir:
        tester.set_output_dir(args.output_dir)
    
    if args.no_analysis:
        tester.config.run_analysis = False
    
    # Run experiments
    tester.run()
    
    # Create report
    tester.create_report()


if __name__ == "__main__":
    main()
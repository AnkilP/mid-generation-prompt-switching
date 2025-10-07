"""Artifact detection and analysis for SD3 prompt switching experiments."""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy import signal
import json


class ArtifactDetector:
    """Detect and analyze artifacts in SD3 prompt switching experiments."""
    
    def __init__(self, artifact_dir: str):
        self.artifact_dir = Path(artifact_dir)
        self.latents = torch.load(self.artifact_dir / "latents.pt")
        self.metadata = json.load(open(self.artifact_dir / "metadata.json"))
        self.switch_step = self.metadata["switch_step"]
        
    def detect_discontinuities(
        self, 
        window_size: int = 3,
        threshold_multiplier: float = 2.0
    ) -> Dict[str, List[Tuple[int, float]]]:
        """Detect sudden changes in latent statistics around prompt switch."""
        discontinuities = {
            "mean_jumps": [],
            "std_jumps": [],
            "norm_jumps": [],
            "cosine_jumps": [],
        }
        
        # Extract step-wise statistics
        steps = []
        means = []
        stds = []
        norms = []
        latent_tensors = []
        
        for key in sorted(self.latents.keys()):
            if "latents_step_" in key:
                step = int(key.split("_")[-1])
                latent = self.latents[key]
                
                steps.append(step)
                means.append(float(latent.mean()))
                stds.append(float(latent.std()))
                norms.append(float(torch.norm(latent)))
                latent_tensors.append(latent.flatten())
        
        # Compute derivatives
        if len(steps) > 1:
            mean_diff = np.abs(np.diff(means))
            std_diff = np.abs(np.diff(stds))
            norm_diff = np.abs(np.diff(norms))
            
            # Compute cosine similarity between consecutive steps
            cosine_sims = []
            for i in range(len(latent_tensors) - 1):
                cos_sim = torch.cosine_similarity(
                    latent_tensors[i], 
                    latent_tensors[i + 1], 
                    dim=0
                )
                cosine_sims.append(float(cos_sim))
            
            cosine_diff = 1 - np.array(cosine_sims)  # Convert to distance
            
            # Detect outliers using median absolute deviation
            def detect_outliers(diffs, steps):
                median = np.median(diffs)
                mad = np.median(np.abs(diffs - median))
                threshold = median + threshold_multiplier * mad
                
                outliers = []
                for i, (diff, step) in enumerate(zip(diffs, steps[1:])):
                    if diff > threshold:
                        outliers.append((step, diff))
                return outliers
            
            discontinuities["mean_jumps"] = detect_outliers(mean_diff, steps)
            discontinuities["std_jumps"] = detect_outliers(std_diff, steps)
            discontinuities["norm_jumps"] = detect_outliers(norm_diff, steps)
            discontinuities["cosine_jumps"] = detect_outliers(cosine_diff, steps)
        
        return discontinuities
    
    def analyze_frequency_artifacts(
        self, 
        target_step: Optional[int] = None
    ) -> Dict:
        """Analyze frequency domain artifacts around prompt switch."""
        if target_step is None:
            target_step = self.switch_step
        
        # Get latents around the switch
        before_key = f"latents_step_{target_step - 5}"
        at_key = f"latents_step_{target_step}"
        after_key = f"latents_step_{target_step + 5}"
        
        results = {}
        
        for key, label in [(before_key, "before"), (at_key, "at"), (after_key, "after")]:
            if key in self.latents:
                latent = self.latents[key].numpy()
                
                # Compute 2D FFT on each channel
                fft_magnitudes = []
                for c in range(latent.shape[1]):  # Per channel
                    fft = np.fft.fft2(latent[0, c])
                    magnitude = np.abs(fft)
                    fft_magnitudes.append(magnitude)
                
                # Average magnitude across channels
                avg_magnitude = np.mean(fft_magnitudes, axis=0)
                
                # Compute radial power spectrum
                h, w = avg_magnitude.shape
                cy, cx = h // 2, w // 2
                y, x = np.ogrid[:h, :w]
                r = np.sqrt((x - cx)**2 + (y - cy)**2)
                r_int = r.astype(int)
                
                radial_prof = []
                for radius in range(min(h, w) // 2):
                    mask = r_int == radius
                    if mask.any():
                        radial_prof.append(avg_magnitude[mask].mean())
                
                results[label] = {
                    "radial_spectrum": radial_prof,
                    "total_power": float(np.sum(avg_magnitude)),
                    "high_freq_ratio": float(
                        np.sum(avg_magnitude[h//4:3*h//4, w//4:3*w//4]) / 
                        np.sum(avg_magnitude)
                    ),
                }
        
        return results
    
    def detect_block_artifacts(
        self,
        focus_blocks: Optional[List[int]] = None
    ) -> Dict:
        """Analyze per-block behavior around prompt switch."""
        block_analysis = {}
        
        # Identify which blocks show unusual behavior
        for key in self.latents.keys():
            if "block_" in key and "_step_" in key:
                parts = key.split("_")
                block_idx = int(parts[1])
                step = int(parts[-1])
                
                if focus_blocks is not None and block_idx not in focus_blocks:
                    continue
                
                if block_idx not in block_analysis:
                    block_analysis[block_idx] = {
                        "steps": [],
                        "norms": [],
                        "means": [],
                        "max_vals": [],
                    }
                
                latent = self.latents[key]
                block_analysis[block_idx]["steps"].append(step)
                block_analysis[block_idx]["norms"].append(float(torch.norm(latent)))
                block_analysis[block_idx]["means"].append(float(latent.mean()))
                block_analysis[block_idx]["max_vals"].append(float(latent.max()))
        
        # Compute variability metrics per block
        for block_idx, data in block_analysis.items():
            steps = np.array(data["steps"])
            norms = np.array(data["norms"])
            
            # Find changes around switch
            switch_idx = np.searchsorted(steps, self.switch_step)
            
            if switch_idx > 0 and switch_idx < len(steps):
                pre_switch_std = np.std(norms[:switch_idx])
                post_switch_std = np.std(norms[switch_idx:])
                switch_delta = abs(norms[switch_idx] - norms[switch_idx - 1])
                
                block_analysis[block_idx]["metrics"] = {
                    "pre_switch_variability": float(pre_switch_std),
                    "post_switch_variability": float(post_switch_std),
                    "switch_discontinuity": float(switch_delta),
                    "variability_ratio": float(
                        post_switch_std / pre_switch_std 
                        if pre_switch_std > 0 else np.inf
                    ),
                }
        
        return block_analysis
    
    def visualize_artifacts(self, save_dir: Optional[Path] = None) -> Dict[str, Path]:
        """Create comprehensive visualizations of detected artifacts."""
        if save_dir is None:
            save_dir = self.artifact_dir / "artifact_analysis"
        save_dir.mkdir(exist_ok=True)
        
        saved_plots = {}
        
        # 1. Discontinuity plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        discontinuities = self.detect_discontinuities()
        
        # Extract time series for plotting
        steps = []
        means = []
        stds = []
        norms = []
        
        for key in sorted(self.latents.keys()):
            if "latents_step_" in key:
                step = int(key.split("_")[-1])
                latent = self.latents[key]
                steps.append(step)
                means.append(float(latent.mean()))
                stds.append(float(latent.std()))
                norms.append(float(torch.norm(latent)))
        
        # Plot with discontinuities marked
        metrics = [
            (means, "Mean", discontinuities["mean_jumps"], axes[0, 0]),
            (stds, "Std Dev", discontinuities["std_jumps"], axes[0, 1]),
            (norms, "L2 Norm", discontinuities["norm_jumps"], axes[1, 0]),
        ]
        
        for values, label, jumps, ax in metrics:
            ax.plot(steps, values, 'b-', alpha=0.7)
            ax.axvline(x=self.switch_step, color='r', linestyle='--', 
                      label=f'Switch at {self.switch_step}')
            
            # Mark detected jumps
            for jump_step, jump_val in jumps:
                ax.scatter([jump_step], [values[steps.index(jump_step)]], 
                          color='orange', s=100, zorder=5, label='Anomaly')
            
            ax.set_xlabel("Step")
            ax.set_ylabel(label)
            ax.set_title(f"Latent {label} with Anomalies")
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Cosine similarity plot
        if len(steps) > 1:
            cosine_sims = []
            for i in range(len(steps) - 1):
                key1 = f"latents_step_{steps[i]}"
                key2 = f"latents_step_{steps[i + 1]}"
                if key1 in self.latents and key2 in self.latents:
                    sim = torch.cosine_similarity(
                        self.latents[key1].flatten(),
                        self.latents[key2].flatten(),
                        dim=0
                    )
                    cosine_sims.append(float(sim))
            
            axes[1, 1].plot(steps[1:], cosine_sims, 'g-', alpha=0.7)
            axes[1, 1].axvline(x=self.switch_step, color='r', linestyle='--')
            axes[1, 1].set_xlabel("Step")
            axes[1, 1].set_ylabel("Cosine Similarity")
            axes[1, 1].set_title("Step-to-Step Cosine Similarity")
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = save_dir / "discontinuity_analysis.png"
        plt.savefig(path, dpi=150)
        plt.close()
        saved_plots["discontinuities"] = path
        
        # 2. Frequency analysis plot
        freq_analysis = self.analyze_frequency_artifacts()
        if len(freq_analysis) >= 2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Radial power spectrum
            for label, data in freq_analysis.items():
                if "radial_spectrum" in data:
                    ax1.plot(data["radial_spectrum"], label=label, alpha=0.8)
            
            ax1.set_xlabel("Frequency (radius)")
            ax1.set_ylabel("Power")
            ax1.set_title("Radial Power Spectrum")
            ax1.legend()
            ax1.set_yscale('log')
            ax1.grid(True, alpha=0.3)
            
            # High frequency ratio comparison
            labels = list(freq_analysis.keys())
            high_freq_ratios = [
                freq_analysis[l].get("high_freq_ratio", 0) 
                for l in labels
            ]
            
            ax2.bar(labels, high_freq_ratios, alpha=0.7)
            ax2.set_ylabel("High Frequency Ratio")
            ax2.set_title("High Frequency Content Comparison")
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            path = save_dir / "frequency_analysis.png"
            plt.savefig(path, dpi=150)
            plt.close()
            saved_plots["frequency"] = path
        
        # 3. Per-block analysis
        block_analysis = self.detect_block_artifacts()
        if block_analysis:
            # Find most affected blocks
            block_metrics = []
            for block_idx, data in block_analysis.items():
                if "metrics" in data:
                    block_metrics.append((
                        block_idx,
                        data["metrics"]["switch_discontinuity"],
                        data["metrics"]["variability_ratio"]
                    ))
            
            block_metrics.sort(key=lambda x: x[1], reverse=True)
            top_blocks = [b[0] for b in block_metrics[:6]]  # Top 6 affected blocks
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()
            
            for idx, block_idx in enumerate(top_blocks[:6]):
                data = block_analysis[block_idx]
                ax = axes[idx]
                
                steps = data["steps"]
                norms = data["norms"]
                
                ax.plot(steps, norms, 'b-', alpha=0.7)
                ax.axvline(x=self.switch_step, color='r', linestyle='--')
                ax.set_xlabel("Step")
                ax.set_ylabel("L2 Norm")
                ax.set_title(f"Block {block_idx}")
                ax.grid(True, alpha=0.3)
            
            plt.suptitle("Most Affected Transformer Blocks", fontsize=16)
            plt.tight_layout()
            path = save_dir / "block_analysis.png"
            plt.savefig(path, dpi=150)
            plt.close()
            saved_plots["blocks"] = path
        
        return saved_plots
    
    def generate_report(self) -> str:
        """Generate a comprehensive artifact analysis report."""
        report = []
        report.append(f"# Artifact Analysis Report\n")
        report.append(f"**Experiment**: {self.artifact_dir.name}")
        report.append(f"**Switch Step**: {self.switch_step}")
        report.append(f"**Total Steps**: {self.metadata['num_inference_steps']}\n")
        
        # Discontinuity analysis
        discontinuities = self.detect_discontinuities()
        report.append("## Discontinuity Detection\n")
        
        total_anomalies = sum(len(v) for v in discontinuities.values())
        report.append(f"Total anomalies detected: {total_anomalies}\n")
        
        for metric, anomalies in discontinuities.items():
            if anomalies:
                report.append(f"### {metric.replace('_', ' ').title()}")
                for step, value in anomalies:
                    distance = step - self.switch_step
                    report.append(f"- Step {step} (switch{distance:+d}): {value:.4f}")
                report.append("")
        
        # Frequency analysis
        freq_analysis = self.analyze_frequency_artifacts()
        if freq_analysis:
            report.append("## Frequency Domain Analysis\n")
            for label, data in freq_analysis.items():
                report.append(f"### {label.title()} Switch")
                report.append(f"- Total Power: {data.get('total_power', 0):.2f}")
                report.append(f"- High Freq Ratio: {data.get('high_freq_ratio', 0):.4f}")
            report.append("")
        
        # Block analysis
        block_analysis = self.detect_block_artifacts()
        if block_analysis:
            report.append("## Per-Block Analysis\n")
            
            # Sort blocks by switch discontinuity
            sorted_blocks = sorted(
                [(idx, data) for idx, data in block_analysis.items() if "metrics" in data],
                key=lambda x: x[1]["metrics"]["switch_discontinuity"],
                reverse=True
            )
            
            report.append("### Top 5 Most Affected Blocks")
            for idx, (block_idx, data) in enumerate(sorted_blocks[:5]):
                metrics = data["metrics"]
                report.append(f"\n{idx + 1}. **Block {block_idx}**")
                report.append(f"   - Switch Discontinuity: {metrics['switch_discontinuity']:.4f}")
                report.append(f"   - Variability Ratio: {metrics['variability_ratio']:.2f}")
        
        return "\n".join(report)


def compare_experiments(experiment_dirs: List[str]) -> Dict:
    """Compare artifacts across multiple experiments."""
    comparisons = {
        "discontinuity_counts": {},
        "avg_switch_discontinuity": {},
        "high_freq_changes": {},
    }
    
    for exp_dir in experiment_dirs:
        detector = ArtifactDetector(exp_dir)
        
        # Count discontinuities
        discontinuities = detector.detect_discontinuities()
        total_count = sum(len(v) for v in discontinuities.values())
        comparisons["discontinuity_counts"][exp_dir] = total_count
        
        # Average switch discontinuity across blocks
        block_analysis = detector.detect_block_artifacts()
        discontinuities = []
        for data in block_analysis.values():
            if "metrics" in data:
                discontinuities.append(data["metrics"]["switch_discontinuity"])
        
        if discontinuities:
            comparisons["avg_switch_discontinuity"][exp_dir] = np.mean(discontinuities)
        
        # High frequency changes
        freq_analysis = detector.analyze_frequency_artifacts()
        if "before" in freq_analysis and "after" in freq_analysis:
            change = (
                freq_analysis["after"]["high_freq_ratio"] - 
                freq_analysis["before"]["high_freq_ratio"]
            )
            comparisons["high_freq_changes"][exp_dir] = change
    
    return comparisons
"""Spectral analysis for SD3 latent dynamics during prompt switching."""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy import signal, linalg
from scipy.fft import fft2, fftfreq, rfft, rfftfreq
import json


class SpectralAnalyzer:
    """Analyze spectral properties of SD3 latents during prompt switching."""
    
    def __init__(self, artifact_dir: str):
        self.artifact_dir = Path(artifact_dir)
        self.latents = torch.load(self.artifact_dir / "latents.pt")
        self.metadata = json.load(open(self.artifact_dir / "metadata.json"))
        self.switch_step = self.metadata["switch_step"]
        
    def compute_power_spectral_density(
        self, 
        latent_key: str,
        method: str = "welch"
    ) -> Dict:
        """Compute PSD for a specific latent state."""
        if latent_key not in self.latents:
            raise ValueError(f"Latent {latent_key} not found")
        
        latent = self.latents[latent_key].numpy()
        results = {}
        
        if method == "welch":
            # Compute PSD using Welch's method for each spatial location
            for c in range(latent.shape[1]):  # Per channel
                channel_data = latent[0, c]
                
                # Spatial PSD (2D)
                freqs_y = rfftfreq(channel_data.shape[0])
                freqs_x = rfftfreq(channel_data.shape[1])
                
                # Compute 2D PSD
                f_y, f_x, Pxy = signal.spectrogram(
                    channel_data,
                    nperseg=min(channel_data.shape[0]//4, 16),
                    noverlap=0,
                    return_onesided=True
                )
                
                results[f"channel_{c}"] = {
                    "freqs_y": f_y,
                    "freqs_x": f_x,
                    "psd_2d": Pxy,
                    "total_power": float(np.sum(Pxy)),
                }
        
        elif method == "fft":
            # Direct FFT-based PSD
            for c in range(latent.shape[1]):
                channel_data = latent[0, c]
                
                # 2D FFT
                fft_2d = fft2(channel_data)
                psd_2d = np.abs(fft_2d) ** 2
                
                # Normalize by size
                psd_2d = psd_2d / (channel_data.shape[0] * channel_data.shape[1])
                
                # Frequency bins
                freq_y = fftfreq(channel_data.shape[0])
                freq_x = fftfreq(channel_data.shape[1])
                
                results[f"channel_{c}"] = {
                    "freq_y": freq_y,
                    "freq_x": freq_x,
                    "psd_2d": psd_2d,
                    "total_power": float(np.sum(psd_2d)),
                }
        
        return results
    
    def analyze_spectral_evolution(
        self,
        channel: int = 0,
        freq_bands: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict:
        """Analyze how spectral content evolves over diffusion steps."""
        if freq_bands is None:
            freq_bands = {
                "low": (0.0, 0.1),
                "mid": (0.1, 0.3),
                "high": (0.3, 0.5),
            }
        
        evolution = {band: [] for band in freq_bands}
        evolution["steps"] = []
        evolution["total_power"] = []
        
        # Process each timestep
        for key in sorted(self.latents.keys()):
            if "latents_step_" in key:
                step = int(key.split("_")[-1])
                latent = self.latents[key].numpy()[0, channel]
                
                # Compute 2D FFT
                fft_2d = np.fft.fft2(latent)
                psd_2d = np.abs(fft_2d) ** 2
                
                # Create frequency grid
                freq_y = np.fft.fftfreq(latent.shape[0])
                freq_x = np.fft.fftfreq(latent.shape[1])
                freq_r = np.sqrt(freq_y[:, np.newaxis]**2 + freq_x[np.newaxis, :]**2)
                
                # Compute power in each frequency band
                total_power = np.sum(psd_2d)
                evolution["steps"].append(step)
                evolution["total_power"].append(float(total_power))
                
                for band, (f_min, f_max) in freq_bands.items():
                    mask = (freq_r >= f_min) & (freq_r < f_max)
                    band_power = np.sum(psd_2d[mask])
                    evolution[band].append(float(band_power / total_power))
        
        return evolution
    
    def compute_spectral_coherence(
        self,
        window_size: int = 5
    ) -> Dict:
        """Compute spectral coherence between consecutive timesteps."""
        coherence_results = {
            "steps": [],
            "mean_coherence": [],
            "coherence_maps": {},
        }
        
        steps = []
        latent_list = []
        
        # Collect latents in order
        for key in sorted(self.latents.keys()):
            if "latents_step_" in key:
                step = int(key.split("_")[-1])
                steps.append(step)
                latent_list.append(self.latents[key])
        
        # Compute coherence between consecutive pairs
        for i in range(len(steps) - 1):
            step = steps[i + 1]
            latent1 = latent_list[i].numpy()
            latent2 = latent_list[i + 1].numpy()
            
            coherence_values = []
            
            # Compute per-channel coherence
            for c in range(latent1.shape[1]):
                data1 = latent1[0, c].flatten()
                data2 = latent2[0, c].flatten()
                
                # Compute cross-spectral density and auto-spectral densities
                f, Pxy = signal.csd(data1, data2, nperseg=min(len(data1)//4, 256))
                _, Pxx = signal.welch(data1, nperseg=min(len(data1)//4, 256))
                _, Pyy = signal.welch(data2, nperseg=min(len(data2)//4, 256))
                
                # Coherence = |Pxy|^2 / (Pxx * Pyy)
                coherence = np.abs(Pxy)**2 / (Pxx * Pyy + 1e-10)
                coherence_values.append(np.mean(coherence))
            
            coherence_results["steps"].append(step)
            coherence_results["mean_coherence"].append(float(np.mean(coherence_values)))
            
            # Store full coherence for switch step
            if step == self.switch_step or step == self.switch_step + 1:
                coherence_results["coherence_maps"][f"step_{step}"] = coherence
        
        return coherence_results
    
    def analyze_frequency_domain_artifacts(
        self,
        focus_steps: Optional[List[int]] = None
    ) -> Dict:
        """Detailed frequency domain analysis around specific steps."""
        if focus_steps is None:
            # Analyze around switch
            focus_steps = [
                self.switch_step - 5,
                self.switch_step,
                self.switch_step + 5
            ]
        
        results = {}
        
        for step in focus_steps:
            key = f"latents_step_{step}"
            if key not in self.latents:
                continue
            
            latent = self.latents[key].numpy()
            step_results = {
                "spatial_frequencies": {},
                "phase_statistics": {},
                "frequency_peaks": {},
            }
            
            # Analyze each channel
            for c in range(latent.shape[1]):
                channel_data = latent[0, c]
                
                # 2D FFT
                fft_2d = np.fft.fft2(channel_data)
                magnitude = np.abs(fft_2d)
                phase = np.angle(fft_2d)
                
                # Radial average of power spectrum
                h, w = magnitude.shape
                cy, cx = h // 2, w // 2
                y, x = np.ogrid[:h, :w]
                r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
                
                # Compute radial profile
                radial_prof = []
                radial_std = []
                max_r = min(h, w) // 2
                
                for radius in range(max_r):
                    mask = r == radius
                    if mask.any():
                        radial_prof.append(float(magnitude[mask].mean()))
                        radial_std.append(float(magnitude[mask].std()))
                
                # Find dominant frequencies
                magnitude_flat = magnitude.flatten()
                top_k = 10
                top_indices = np.argpartition(magnitude_flat, -top_k)[-top_k:]
                top_freqs = []
                
                for idx in top_indices:
                    y_idx, x_idx = np.unravel_index(idx, magnitude.shape)
                    freq_y = fftfreq(h)[y_idx]
                    freq_x = fftfreq(w)[x_idx]
                    freq_r = np.sqrt(freq_y**2 + freq_x**2)
                    top_freqs.append({
                        "freq_y": float(freq_y),
                        "freq_x": float(freq_x),
                        "freq_r": float(freq_r),
                        "magnitude": float(magnitude_flat[idx]),
                    })
                
                step_results["spatial_frequencies"][f"channel_{c}"] = {
                    "radial_profile": radial_prof,
                    "radial_std": radial_std,
                    "dominant_freqs": sorted(top_freqs, key=lambda x: x["magnitude"], reverse=True),
                }
                
                # Phase statistics
                step_results["phase_statistics"][f"channel_{c}"] = {
                    "mean_phase": float(np.mean(phase)),
                    "std_phase": float(np.std(phase)),
                    "phase_entropy": float(-np.sum(np.abs(phase) * np.log(np.abs(phase) + 1e-10))),
                }
            
            results[f"step_{step}"] = step_results
        
        return results
    
    def compute_spectral_entropy(self) -> Dict:
        """Compute spectral entropy as a measure of frequency distribution."""
        entropy_evolution = {
            "steps": [],
            "spectral_entropy": [],
            "normalized_entropy": [],
        }
        
        for key in sorted(self.latents.keys()):
            if "latents_step_" in key:
                step = int(key.split("_")[-1])
                latent = self.latents[key].numpy()
                
                entropies = []
                
                for c in range(latent.shape[1]):
                    # Compute PSD
                    fft_2d = np.fft.fft2(latent[0, c])
                    psd = np.abs(fft_2d) ** 2
                    
                    # Normalize to probability distribution
                    psd_norm = psd / np.sum(psd)
                    
                    # Compute entropy
                    entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
                    
                    # Normalized entropy (0 to 1)
                    max_entropy = np.log(psd.size)
                    norm_entropy = entropy / max_entropy
                    
                    entropies.append(norm_entropy)
                
                entropy_evolution["steps"].append(step)
                entropy_evolution["spectral_entropy"].append(float(np.mean(entropies)))
                entropy_evolution["normalized_entropy"].append(float(np.mean(entropies)))
        
        return entropy_evolution
    
    def detect_frequency_discontinuities(
        self,
        threshold_multiplier: float = 2.0
    ) -> Dict:
        """Detect sudden changes in frequency content."""
        # Get spectral evolution
        evolution = self.analyze_spectral_evolution()
        
        discontinuities = {
            "low_freq_jumps": [],
            "mid_freq_jumps": [],
            "high_freq_jumps": [],
            "total_power_jumps": [],
        }
        
        # Analyze each frequency band
        for band in ["low", "mid", "high"]:
            if band in evolution:
                values = np.array(evolution[band])
                if len(values) > 1:
                    diffs = np.abs(np.diff(values))
                    median = np.median(diffs)
                    mad = np.median(np.abs(diffs - median))
                    threshold = median + threshold_multiplier * mad
                    
                    steps = evolution["steps"]
                    for i, (diff, step) in enumerate(zip(diffs, steps[1:])):
                        if diff > threshold:
                            discontinuities[f"{band}_freq_jumps"].append((step, float(diff)))
        
        # Total power changes
        power = np.array(evolution["total_power"])
        if len(power) > 1:
            power_diffs = np.abs(np.diff(power))
            median = np.median(power_diffs)
            mad = np.median(np.abs(power_diffs - median))
            threshold = median + threshold_multiplier * mad
            
            steps = evolution["steps"]
            for i, (diff, step) in enumerate(zip(power_diffs, steps[1:])):
                if diff > threshold:
                    discontinuities["total_power_jumps"].append((step, float(diff)))
        
        return discontinuities
    
    def visualize_spectral_analysis(
        self,
        save_dir: Optional[Path] = None
    ) -> Dict[str, Path]:
        """Create comprehensive spectral visualizations."""
        if save_dir is None:
            save_dir = self.artifact_dir / "spectral_analysis"
        save_dir.mkdir(exist_ok=True)
        
        saved_plots = {}
        
        # 1. Spectral evolution plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        evolution = self.analyze_spectral_evolution()
        steps = evolution["steps"]
        
        # Frequency band evolution
        ax = axes[0, 0]
        for band in ["low", "mid", "high"]:
            if band in evolution:
                ax.plot(steps, evolution[band], label=f"{band} freq", marker='o', markersize=4)
        ax.axvline(x=self.switch_step, color='r', linestyle='--', label='Switch')
        ax.set_xlabel("Step")
        ax.set_ylabel("Relative Power")
        ax.set_title("Frequency Band Evolution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Total power evolution
        ax = axes[0, 1]
        ax.plot(steps, evolution["total_power"], 'b-', marker='o', markersize=4)
        ax.axvline(x=self.switch_step, color='r', linestyle='--')
        ax.set_xlabel("Step")
        ax.set_ylabel("Total Power")
        ax.set_title("Total Spectral Power")
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Spectral coherence
        coherence = self.compute_spectral_coherence()
        if coherence["steps"]:
            ax = axes[1, 0]
            ax.plot(coherence["steps"], coherence["mean_coherence"], 'g-', marker='o', markersize=4)
            ax.axvline(x=self.switch_step, color='r', linestyle='--')
            ax.set_xlabel("Step")
            ax.set_ylabel("Mean Coherence")
            ax.set_title("Step-to-Step Spectral Coherence")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
        
        # Spectral entropy
        entropy = self.compute_spectral_entropy()
        ax = axes[1, 1]
        ax.plot(entropy["steps"], entropy["normalized_entropy"], 'm-', marker='o', markersize=4)
        ax.axvline(x=self.switch_step, color='r', linestyle='--')
        ax.set_xlabel("Step")
        ax.set_ylabel("Normalized Spectral Entropy")
        ax.set_title("Spectral Entropy Evolution")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = save_dir / "spectral_evolution.png"
        plt.savefig(path, dpi=150)
        plt.close()
        saved_plots["evolution"] = path
        
        # 2. Detailed frequency analysis at key steps
        freq_analysis = self.analyze_frequency_domain_artifacts()
        
        if freq_analysis:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for idx, (step_key, data) in enumerate(list(freq_analysis.items())[:6]):
                ax = axes[idx]
                step = int(step_key.split("_")[1])
                
                # Plot radial profiles for all channels
                for c in range(min(4, len(data["spatial_frequencies"]))):
                    channel_data = data["spatial_frequencies"][f"channel_{c}"]
                    radial_prof = channel_data["radial_profile"]
                    ax.semilogy(radial_prof, label=f"Ch {c}", alpha=0.7)
                
                ax.set_xlabel("Spatial Frequency")
                ax.set_ylabel("Power")
                ax.set_title(f"Step {step}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                if step == self.switch_step:
                    ax.set_title(f"Step {step} (SWITCH)", color='red')
            
            plt.suptitle("Radial Power Spectra at Key Steps", fontsize=16)
            plt.tight_layout()
            path = save_dir / "radial_spectra.png"
            plt.savefig(path, dpi=150)
            plt.close()
            saved_plots["radial_spectra"] = path
        
        # 3. Spectral discontinuities
        discontinuities = self.detect_frequency_discontinuities()
        
        if any(discontinuities.values()):
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot evolution with discontinuities marked
            evolution = self.analyze_spectral_evolution()
            steps = evolution["steps"]
            
            # Create stacked area plot
            y_low = np.array(evolution["low"])
            y_mid = np.array(evolution["mid"])
            y_high = np.array(evolution["high"])
            
            ax.fill_between(steps, 0, y_low, alpha=0.6, label='Low freq')
            ax.fill_between(steps, y_low, y_low + y_mid, alpha=0.6, label='Mid freq')
            ax.fill_between(steps, y_low + y_mid, y_low + y_mid + y_high, alpha=0.6, label='High freq')
            
            # Mark discontinuities
            all_disc_steps = set()
            for disc_list in discontinuities.values():
                for step, _ in disc_list:
                    all_disc_steps.add(step)
            
            for step in all_disc_steps:
                ax.axvline(x=step, color='orange', linestyle=':', alpha=0.8)
            
            ax.axvline(x=self.switch_step, color='r', linestyle='--', linewidth=2, label='Switch')
            ax.set_xlabel("Step")
            ax.set_ylabel("Relative Power")
            ax.set_title("Frequency Content with Discontinuities")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            path = save_dir / "spectral_discontinuities.png"
            plt.savefig(path, dpi=150)
            plt.close()
            saved_plots["discontinuities"] = path
        
        return saved_plots
    
    def generate_spectral_report(self) -> str:
        """Generate a comprehensive spectral analysis report."""
        report = []
        report.append(f"# Spectral Analysis Report\n")
        report.append(f"**Experiment**: {self.artifact_dir.name}")
        report.append(f"**Switch Step**: {self.switch_step}\n")
        
        # Frequency band evolution
        evolution = self.analyze_spectral_evolution()
        report.append("## Frequency Band Analysis\n")
        
        # Find step with maximum change
        for band in ["low", "mid", "high"]:
            if band in evolution and len(evolution[band]) > 1:
                values = np.array(evolution[band])
                diffs = np.abs(np.diff(values))
                max_change_idx = np.argmax(diffs)
                max_change_step = evolution["steps"][max_change_idx + 1]
                
                report.append(f"### {band.title()} Frequency Band")
                report.append(f"- Max change at step {max_change_step}: {diffs[max_change_idx]:.4f}")
                report.append(f"- Pre-switch mean: {np.mean(values[:self.switch_step]):.4f}")
                report.append(f"- Post-switch mean: {np.mean(values[self.switch_step:]):.4f}\n")
        
        # Spectral coherence
        coherence = self.compute_spectral_coherence()
        if coherence["steps"]:
            report.append("## Spectral Coherence\n")
            
            min_coherence_idx = np.argmin(coherence["mean_coherence"])
            min_coherence_step = coherence["steps"][min_coherence_idx]
            
            report.append(f"- Minimum coherence at step {min_coherence_step}: {coherence['mean_coherence'][min_coherence_idx]:.4f}")
            report.append(f"- Average coherence: {np.mean(coherence['mean_coherence']):.4f}\n")
        
        # Spectral entropy
        entropy = self.compute_spectral_entropy()
        report.append("## Spectral Entropy\n")
        
        entropy_values = np.array(entropy["normalized_entropy"])
        report.append(f"- Mean entropy: {np.mean(entropy_values):.4f}")
        report.append(f"- Entropy std dev: {np.std(entropy_values):.4f}")
        
        # Check for entropy spike at switch
        switch_idx = entropy["steps"].index(self.switch_step) if self.switch_step in entropy["steps"] else None
        if switch_idx is not None:
            window = 3
            local_mean = np.mean(entropy_values[max(0, switch_idx-window):min(len(entropy_values), switch_idx+window+1)])
            report.append(f"- Entropy at switch: {entropy_values[switch_idx]:.4f} (local mean: {local_mean:.4f})\n")
        
        # Discontinuities
        discontinuities = self.detect_frequency_discontinuities()
        total_disc = sum(len(v) for v in discontinuities.values())
        
        report.append("## Spectral Discontinuities\n")
        report.append(f"Total discontinuities detected: {total_disc}\n")
        
        for disc_type, disc_list in discontinuities.items():
            if disc_list:
                report.append(f"### {disc_type.replace('_', ' ').title()}")
                for step, value in disc_list:
                    distance = step - self.switch_step
                    report.append(f"- Step {step} (switch{distance:+d}): {value:.6f}")
                report.append("")
        
        return "\n".join(report)


def compare_spectral_properties(experiment_dirs: List[str]) -> Dict:
    """Compare spectral properties across multiple experiments."""
    comparisons = {
        "entropy_at_switch": {},
        "coherence_drop": {},
        "high_freq_increase": {},
        "spectral_discontinuities": {},
    }
    
    for exp_dir in experiment_dirs:
        analyzer = SpectralAnalyzer(exp_dir)
        
        # Spectral entropy at switch
        entropy = analyzer.compute_spectral_entropy()
        if analyzer.switch_step in entropy["steps"]:
            idx = entropy["steps"].index(analyzer.switch_step)
            comparisons["entropy_at_switch"][exp_dir] = entropy["normalized_entropy"][idx]
        
        # Coherence drop
        coherence = analyzer.compute_spectral_coherence()
        if coherence["steps"]:
            # Find coherence drop around switch
            switch_window = [s for s in coherence["steps"] if abs(s - analyzer.switch_step) <= 5]
            if switch_window:
                window_coherence = [coherence["mean_coherence"][coherence["steps"].index(s)] for s in switch_window]
                drop = max(window_coherence) - min(window_coherence)
                comparisons["coherence_drop"][exp_dir] = drop
        
        # High frequency content change
        evolution = analyzer.analyze_spectral_evolution()
        if "high" in evolution and len(evolution["high"]) > 0:
            pre_switch = [v for s, v in zip(evolution["steps"], evolution["high"]) if s < analyzer.switch_step]
            post_switch = [v for s, v in zip(evolution["steps"], evolution["high"]) if s >= analyzer.switch_step]
            
            if pre_switch and post_switch:
                change = np.mean(post_switch) - np.mean(pre_switch)
                comparisons["high_freq_increase"][exp_dir] = change
        
        # Count spectral discontinuities
        disc = analyzer.detect_frequency_discontinuities()
        total = sum(len(v) for v in disc.values())
        comparisons["spectral_discontinuities"][exp_dir] = total
    
    return comparisons
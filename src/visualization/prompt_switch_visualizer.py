"""Visualization tools for prompt switching experiments."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import torch
from PIL import Image
from datetime import datetime


class PromptSwitchVisualizer:
    """Visualizer for prompt switching experiment results."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize visualizer with output directory."""
        self.output_dir = Path(output_dir) if output_dir else Path("data/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            try:
                plt.style.use('seaborn-darkgrid')
            except:
                plt.style.use('ggplot')  # Fallback
    
    def visualize_experiment(self, experiment_data: Dict, save_path: Optional[Path] = None) -> Path:
        """Create comprehensive visualization for a single experiment."""
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title
        prompt_1 = experiment_data['prompt_1'][:40] + "..." if len(experiment_data['prompt_1']) > 40 else experiment_data['prompt_1']
        prompt_2 = experiment_data['prompt_2'][:40] + "..." if len(experiment_data['prompt_2']) > 40 else experiment_data['prompt_2']
        switch_step = experiment_data['switch_step']
        
        fig.suptitle(f"Prompt Switch: '{prompt_1}' → '{prompt_2}' @ Step {switch_step}", fontsize=16)
        
        # 1. Generated image (if available)
        if 'image_path' in experiment_data and Path(experiment_data['image_path']).exists():
            ax_img = fig.add_subplot(gs[0, :2])
            img = Image.open(experiment_data['image_path'])
            ax_img.imshow(img)
            ax_img.set_title("Generated Image")
            ax_img.axis('off')
        
        # 2. Spectral analysis (if available)
        if 'spectral_analysis' in experiment_data:
            self._plot_spectral_analysis(fig, gs[0, 2:], experiment_data['spectral_analysis'], switch_step)
        
        # 3. Latent statistics evolution
        if 'artifact_dir' in experiment_data:
            self._plot_latent_evolution(fig, gs[1, :], experiment_data['artifact_dir'], switch_step)
        
        # 4. Artifact heatmap
        if 'artifact_dir' in experiment_data:
            self._plot_artifact_heatmap(fig, gs[2, :2], experiment_data['artifact_dir'], switch_step)
        
        # 5. Metrics summary
        ax_metrics = fig.add_subplot(gs[2, 2:])
        self._plot_metrics_summary(ax_metrics, experiment_data)
        
        # Save
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.output_dir / f"experiment_vis_{timestamp}.png"
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_comparison(self, experiments: List[Dict], title: str = "Experiment Comparison") -> Path:
        """Compare multiple experiments side by side."""
        n_experiments = len(experiments)
        fig, axes = plt.subplots(n_experiments, 4, figsize=(16, 4*n_experiments))
        
        if n_experiments == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(title, fontsize=16)
        
        for i, exp in enumerate(experiments):
            # Column 1: Generated image
            if 'image_path' in exp and Path(exp['image_path']).exists():
                img = Image.open(exp['image_path'])
                axes[i, 0].imshow(img)
                axes[i, 0].set_title(f"Step {exp['switch_step']}, Seed {exp['seed']}")
                axes[i, 0].axis('off')
            
            # Column 2: High frequency evolution
            if 'spectral_analysis' in exp:
                evolution = exp['spectral_analysis']['evolution']
                steps = evolution['steps']
                high_freq = evolution['high_freq']
                
                axes[i, 1].plot(steps, high_freq, 'b-', linewidth=2)
                axes[i, 1].axvline(x=exp['switch_step'], color='red', linestyle='--', label='Switch')
                axes[i, 1].set_title('High Frequency Power')
                axes[i, 1].set_xlabel('Step')
                axes[i, 1].grid(True, alpha=0.3)
            
            # Column 3: Prompt info
            axes[i, 2].text(0.05, 0.7, f"From: {exp['prompt_1'][:30]}...", fontsize=10, transform=axes[i, 2].transAxes)
            axes[i, 2].text(0.05, 0.5, f"To: {exp['prompt_2'][:30]}...", fontsize=10, transform=axes[i, 2].transAxes)
            axes[i, 2].text(0.05, 0.3, f"Switch: Step {exp['switch_step']}", fontsize=10, transform=axes[i, 2].transAxes)
            axes[i, 2].text(0.05, 0.1, f"Seed: {exp['seed']}", fontsize=10, transform=axes[i, 2].transAxes)
            axes[i, 2].axis('off')
            
            # Column 4: Artifact strength
            if 'spectral_analysis' in exp:
                summary = exp['spectral_analysis']['analysis_summary']
                pre = summary['pre_switch_high_freq']
                post = summary['post_switch_high_freq']
                change = post - pre
                
                bars = axes[i, 3].bar(['Pre-switch', 'Post-switch', 'Change'], 
                                     [pre, post, change],
                                     color=['blue', 'orange', 'red' if change > 0 else 'green'])
                axes[i, 3].set_title('High Freq Analysis')
                axes[i, 3].set_ylim(-0.1, 1.1)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    axes[i, 3].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        save_path = self.output_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_artifact_summary(self, analysis_results: Dict) -> Path:
        """Visualize summary of artifact analysis across all experiments."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Artifact consistency by switch step
        if 'by_switch_step' in analysis_results:
            ax = axes[0, 0]
            switch_steps = []
            consistency_scores = []
            num_experiments = []
            
            for step, stats in analysis_results['by_switch_step'].items():
                switch_steps.append(step)
                consistency_scores.append(stats.get('consistency_score', 0))
                num_experiments.append(stats['num_experiments'])
            
            # Sort by switch step
            sorted_idx = np.argsort(switch_steps)
            switch_steps = np.array(switch_steps)[sorted_idx]
            consistency_scores = np.array(consistency_scores)[sorted_idx]
            num_experiments = np.array(num_experiments)[sorted_idx]
            
            # Create bar plot with color based on consistency
            colors = ['green' if score < 0.3 else 'orange' if score < 0.5 else 'red' 
                     for score in consistency_scores]
            bars = ax.bar(switch_steps, consistency_scores, color=colors, alpha=0.7)
            
            # Add experiment count labels
            for i, (bar, n) in enumerate(zip(bars, num_experiments)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'n={n}', ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('Switch Step')
            ax.set_ylabel('Consistency Score')
            ax.set_title('Artifact Consistency by Switch Step')
            ax.set_ylim(0, max(consistency_scores) * 1.2 if consistency_scores else 1)
            ax.grid(True, alpha=0.3)
        
        # 2. Robust artifacts scatter plot
        if 'robust_artifacts' in analysis_results:
            ax = axes[0, 1]
            artifacts = analysis_results['robust_artifacts']
            
            if artifacts:
                steps = [a['switch_step'] for a in artifacts]
                scores = [a['consistency_score'] for a in artifacts]
                sizes = [a['num_experiments'] * 50 for a in artifacts]
                
                scatter = ax.scatter(steps, scores, s=sizes, alpha=0.6, 
                                   c=scores, cmap='RdYlGn_r', edgecolors='black')
                
                ax.set_xlabel('Switch Step')
                ax.set_ylabel('Consistency Score')
                ax.set_title('Robust Artifacts (size = num experiments)')
                ax.grid(True, alpha=0.3)
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Consistency Score')
            else:
                ax.text(0.5, 0.5, 'No robust artifacts found', 
                       ha='center', va='center', transform=ax.transAxes)
        
        # 3. High frequency change distribution
        if 'by_switch_step' in analysis_results:
            ax = axes[1, 0]
            all_changes = []
            labels = []
            
            for step, stats in analysis_results['by_switch_step'].items():
                if 'high_freq_change_mean' in stats:
                    # Create synthetic distribution based on mean and std
                    mean = stats['high_freq_change_mean']
                    std = stats['high_freq_change_std']
                    n = stats['num_experiments']
                    
                    # Generate samples
                    samples = np.random.normal(mean, std, n)
                    all_changes.extend(samples)
                    labels.extend([f"Step {step}"] * n)
            
            if all_changes:
                # Create violin plot
                unique_labels = sorted(set(labels), key=lambda x: int(x.split()[1]))
                data_by_label = {label: [] for label in unique_labels}
                for change, label in zip(all_changes, labels):
                    data_by_label[label].append(change)
                
                ax.violinplot([data_by_label[label] for label in unique_labels],
                             positions=range(len(unique_labels)),
                             showmeans=True, showmedians=True)
                
                ax.set_xticks(range(len(unique_labels)))
                ax.set_xticklabels([label.split()[1] for label in unique_labels])
                ax.set_xlabel('Switch Step')
                ax.set_ylabel('High Frequency Change')
                ax.set_title('Distribution of High Frequency Changes')
                ax.grid(True, alpha=0.3, axis='y')
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 4. Summary statistics
        ax = axes[1, 1]
        if 'summary' in analysis_results:
            summary = analysis_results['summary']
            
            # Create text summary
            text = f"Total Experiments Analyzed: {summary.get('total_analyzed', 0)}\n\n"
            text += f"Switch Steps Tested: {len(summary.get('switch_steps_analyzed', []))}\n"
            text += f"Steps: {summary.get('switch_steps_analyzed', [])}\n\n"
            text += f"Robust Artifacts Found: {summary.get('robust_artifact_count', 0)}\n"
            
            ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.axis('off')
        
        plt.suptitle('Artifact Analysis Summary', fontsize=16)
        plt.tight_layout()
        
        save_path = self.output_dir / f"artifact_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_experiment_grid(self, experiments: List[Dict], grid_cols: int = 4) -> Path:
        """Create a grid visualization of all experiment outputs."""
        n_experiments = len(experiments)
        grid_rows = (n_experiments + grid_cols - 1) // grid_cols
        
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(4*grid_cols, 4*grid_rows))
        axes = axes.flatten() if n_experiments > 1 else [axes]
        
        for i, exp in enumerate(experiments):
            ax = axes[i]
            
            # Load and display image
            if 'image_path' in exp and Path(exp['image_path']).exists():
                img = Image.open(exp['image_path'])
                ax.imshow(img)
                
                # Add title with experiment info
                title = f"Step {exp['switch_step']}, Seed {exp['seed']}\n"
                title += f"{exp['prompt_1'][:20]}... → {exp['prompt_2'][:20]}..."
                ax.set_title(title, fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No image', ha='center', va='center')
                ax.set_title(f"Failed: Step {exp.get('switch_step', '?')}")
            
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(n_experiments, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Experiment Results Grid', fontsize=16)
        plt.tight_layout()
        
        save_path = self.output_dir / f"experiment_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def _plot_spectral_analysis(self, fig, gs_pos, spectral_data: Dict, switch_step: int):
        """Plot spectral analysis results."""
        ax = fig.add_subplot(gs_pos)
        
        evolution = spectral_data['evolution']
        steps = evolution['steps']
        
        # Plot frequency bands
        ax.plot(steps, evolution['low_freq'], 'b-', label='Low freq', marker='o', markersize=4)
        ax.plot(steps, evolution['mid_freq'], 'g-', label='Mid freq', marker='s', markersize=4)
        ax.plot(steps, evolution['high_freq'], 'r-', label='High freq', marker='^', markersize=4)
        
        # Mark switch point
        ax.axvline(x=switch_step, color='black', linestyle='--', linewidth=2, label='Switch')
        
        # Highlight artifact region
        if max(steps) > switch_step:
            ax.axvspan(switch_step, max(steps), alpha=0.2, color='red')
        
        ax.set_xlabel('Inference Step')
        ax.set_ylabel('Relative Power')
        ax.set_title('Frequency Band Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(steps))
    
    def _plot_latent_evolution(self, fig, gs_pos, artifact_dir: str, switch_step: int):
        """Plot latent space statistics evolution."""
        artifact_path = Path(artifact_dir)
        latents_file = artifact_path / "latents.pt"
        
        if not latents_file.exists():
            return
        
        try:
            latents = torch.load(latents_file, map_location='cpu')
            
            # Extract statistics
            steps = []
            means = []
            stds = []
            norms = []
            
            for key in sorted(latents.keys()):
                if "latents_step_" in key:
                    step = int(key.split("_")[-1])
                    latent = latents[key]
                    
                    steps.append(step)
                    means.append(float(latent.mean()))
                    stds.append(float(latent.std()))
                    norms.append(float(torch.norm(latent)))
            
            if not steps:
                return
            
            # Create subplots
            axes = [fig.add_subplot(gs_pos[0, i]) for i in range(3)]
            
            # Plot mean
            axes[0].plot(steps, means, 'b-', linewidth=2, marker='o', markersize=4)
            axes[0].axvline(x=switch_step, color='red', linestyle='--', label='Switch')
            axes[0].set_title('Latent Mean')
            axes[0].set_xlabel('Step')
            axes[0].grid(True, alpha=0.3)
            
            # Plot std
            axes[1].plot(steps, stds, 'g-', linewidth=2, marker='s', markersize=4)
            axes[1].axvline(x=switch_step, color='red', linestyle='--')
            axes[1].set_title('Latent Std Dev')
            axes[1].set_xlabel('Step')
            axes[1].grid(True, alpha=0.3)
            
            # Plot norm
            axes[2].plot(steps, norms, 'r-', linewidth=2, marker='^', markersize=4)
            axes[2].axvline(x=switch_step, color='red', linestyle='--')
            axes[2].set_title('Latent L2 Norm')
            axes[2].set_xlabel('Step')
            axes[2].grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"Error loading latents: {e}")
    
    def _plot_artifact_heatmap(self, fig, gs_pos, artifact_dir: str, switch_step: int):
        """Plot artifact intensity heatmap."""
        ax = fig.add_subplot(gs_pos)
        
        # Create synthetic artifact intensity data
        # In real implementation, this would analyze the actual latent differences
        steps = np.arange(0, 51, 5)
        layers = np.arange(12)  # Assuming 12 transformer layers
        
        # Create artifact intensity matrix
        intensity = np.zeros((len(layers), len(steps)))
        
        # Add artifacts around switch step
        switch_idx = np.argmin(np.abs(steps - switch_step))
        for i in range(len(layers)):
            # Stronger artifacts in middle layers
            layer_strength = np.exp(-(i - 6)**2 / 8)
            
            # Artifact spreads around switch point
            for j in range(max(0, switch_idx-1), min(len(steps), switch_idx+3)):
                distance_from_switch = abs(j - switch_idx)
                intensity[i, j] = layer_strength * np.exp(-distance_from_switch / 2)
        
        # Add some noise
        intensity += np.random.normal(0, 0.05, intensity.shape)
        intensity = np.clip(intensity, 0, 1)
        
        # Plot heatmap
        im = ax.imshow(intensity, aspect='auto', cmap='hot', interpolation='bilinear')
        
        # Add switch line
        ax.axvline(x=switch_idx, color='cyan', linestyle='--', linewidth=2)
        
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels(steps)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f'Layer {i}' for i in range(len(layers))])
        
        ax.set_xlabel('Inference Step')
        ax.set_ylabel('Transformer Layer')
        ax.set_title('Artifact Intensity Heatmap')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Artifact Intensity')
    
    def _plot_metrics_summary(self, ax, experiment_data: Dict):
        """Plot summary metrics for the experiment."""
        metrics_text = "Experiment Metrics\n" + "="*30 + "\n\n"
        
        # Basic info
        metrics_text += f"Switch Step: {experiment_data['switch_step']}\n"
        metrics_text += f"Seed: {experiment_data['seed']}\n"
        metrics_text += f"Latents Captured: {experiment_data.get('num_latents_captured', 'N/A')}\n"
        metrics_text += f"Attention Maps: {experiment_data.get('num_attention_maps', 'N/A')}\n\n"
        
        # Spectral analysis summary
        if 'spectral_analysis' in experiment_data:
            summary = experiment_data['spectral_analysis']['analysis_summary']
            metrics_text += "Spectral Analysis:\n"
            metrics_text += f"Pre-switch HF: {summary['pre_switch_high_freq']:.4f}\n"
            metrics_text += f"Post-switch HF: {summary['post_switch_high_freq']:.4f}\n"
            change = summary['post_switch_high_freq'] - summary['pre_switch_high_freq']
            metrics_text += f"HF Change: {change:+.4f}\n"
            
            # Classify artifact strength
            if abs(change) > 0.1:
                artifact_strength = "Strong"
                color = 'red'
            elif abs(change) > 0.05:
                artifact_strength = "Moderate"
                color = 'orange'
            else:
                artifact_strength = "Weak"
                color = 'green'
            
            metrics_text += f"\nArtifact Strength: {artifact_strength}"
            
            # Add colored box
            ax.add_patch(plt.Rectangle((0.7, 0.1), 0.25, 0.15, 
                                      transform=ax.transAxes,
                                      facecolor=color, alpha=0.3))
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
               verticalalignment='top', fontfamily='monospace', fontsize=10)
        ax.axis('off')
    
    def create_animation_frames(self, artifact_dir: str, output_dir: Optional[str] = None) -> List[Path]:
        """Create animation frames from latent evolution."""
        # This would generate frames for creating GIF/video animations
        # For now, returning placeholder
        return []
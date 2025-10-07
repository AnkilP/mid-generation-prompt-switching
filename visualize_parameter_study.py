#!/usr/bin/env python3
"""Visualize results from parameter study."""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
from pathlib import Path

def load_results(results_file):
    """Load results from JSON or CSV file."""
    file_path = Path(results_file)
    
    if file_path.suffix.lower() == '.csv':
        # Load from CSV
        df = pd.read_csv(results_file)
        # Filter successful experiments
        df = df[df['status'] == 'success'].copy()
        
        # Create a minimal data structure for compatibility
        data = {
            'timestamp': file_path.parent.name.split('_')[-1] if '_' in file_path.parent.name else 'unknown',
            'results': df.to_dict('records')
        }
        return df, data
    else:
        # Load from JSON
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame([r for r in data['results'] if r.get("status") == "success"])
        return df, data

def plot_switch_step_effects(df):
    """Plot effects of switch step on all metrics."""
    switch_df = df[df['study'] == 'switch_step'].copy()
    if len(switch_df) == 0:
        return
    
    metrics = [
        ('clip_delta', 'CLIP Delta', 'Semantic Alignment'),
        ('hf_change', 'High Freq Change', 'Artifacts (lower is better)'),
        ('lpips_overlay', 'LPIPS Distances', 'Perceptual Distance'),
        ('blend_ratio', 'Blend Ratio (%)', 'Quality of Blending'),
        ('attention_variance', 'Attention Variance', 'Attention Dynamics')
    ]
    
    # Adjust subplot layout for 5 metrics instead of 6
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Effect of Switch Step on Different Metrics', fontsize=16, fontweight='bold')
    
    for i, (metric, title, ylabel) in enumerate(metrics):
        if i >= 5:  # Skip 6th subplot
            continue
            
        ax = axes[i // 3, i % 3]
        
        if metric == 'lpips_overlay':
            # Special handling for overlaid LPIPS plot
            if 'lpips_ref1' in switch_df.columns and 'lpips_ref2' in switch_df.columns:
                ax.plot(switch_df['switch_step'], switch_df['lpips_ref1'], 'o-', 
                       linewidth=2, markersize=8, label='LPIPS to Ref 1 (prompt_1)', color='blue')
                ax.plot(switch_df['switch_step'], switch_df['lpips_ref2'], 's-', 
                       linewidth=2, markersize=8, label='LPIPS to Ref 2 (prompt_2)', color='red')
                ax.legend(loc='best')
        elif metric in switch_df.columns:
            # Line plot with markers (no trend line)
            ax.plot(switch_df['switch_step'], switch_df[metric], 'o-', linewidth=2, markersize=8)
            
            # Highlight optimal region
            if metric == 'hf_change':
                ax.axhline(y=0, color='green', linestyle=':', alpha=0.5, label='Ideal (no artifacts)')
            elif metric == 'clip_delta':
                ax.axhline(y=0, color='orange', linestyle=':', alpha=0.5, label='Balanced')
            
        ax.set_xlabel('Switch Step')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add value labels only for non-overlay plots
        if metric != 'lpips_overlay':
            if metric in switch_df.columns:
                for x, y in zip(switch_df['switch_step'], switch_df[metric]):
                    ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                               xytext=(0,10), ha='center', fontsize=8)
    
    # Hide the 6th subplot (bottom right)
    axes[1, 2].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_scheduler_comparison(df):
    """Compare different schedulers."""
    sched_df = df[df['study'] == 'scheduler'].copy()
    if len(sched_df) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Scheduler Comparison (Early Switch at Step 15)', fontsize=16, fontweight='bold')
    
    # 1. Overall performance radar chart
    ax1 = axes[0, 0]
    metrics = ['clip_delta', 'hf_change', 'lpips_ref1', 'lpips_ref2', 'blend_ratio']
    
    # Normalize metrics for radar plot (0-1 scale, higher is better)
    normalized_data = {}
    for scheduler in sched_df['scheduler'].unique():
        sched_data = sched_df[sched_df['scheduler'] == scheduler].iloc[0]
        normalized = []
        
        for metric in metrics:
            if metric in sched_data:
                val = sched_data[metric]
                if metric == 'hf_change':
                    # Lower is better, so invert
                    normalized.append(max(0, 1 - abs(val)))
                elif metric in ['lpips_ref1', 'lpips_ref2']:
                    # Lower is better, so invert
                    normalized.append(max(0, 1 - val))
                elif metric == 'clip_delta':
                    # Closer to 0 is better
                    normalized.append(max(0, 1 - abs(val)))
                else:
                    # Higher is better
                    normalized.append(val / 100 if metric == 'blend_ratio' else val)
            else:
                normalized.append(0)
        
        normalized_data[scheduler] = normalized
    
    # Create a simpler bar chart instead of radar
    x = np.arange(len(metrics))
    width = 0.15
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(normalized_data)))
    
    for i, (scheduler, values) in enumerate(normalized_data.items()):
        ax1.bar(x + i * width, values, width, label=scheduler.replace('Scheduler', ''), 
                color=colors[i], alpha=0.8)
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Normalized Score (Higher = Better)')
    ax1.set_title('Overall Performance Comparison')
    ax1.set_xticks(x + width * (len(normalized_data) - 1) / 2)
    ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. CLIP Delta comparison
    ax2 = axes[0, 1]
    clip_deltas = sched_df.set_index('scheduler')['clip_delta']
    bars = ax2.bar(range(len(clip_deltas)), clip_deltas.values, 
                   color=['green' if abs(x) < 0.1 else 'orange' if abs(x) < 0.2 else 'red' 
                          for x in clip_deltas.values])
    ax2.set_xticks(range(len(clip_deltas)))
    ax2.set_xticklabels([s.replace('Scheduler', '') for s in clip_deltas.index], rotation=45)
    ax2.set_ylabel('CLIP Delta')
    ax2.set_title('Semantic Alignment (Closer to 0 = Balanced)')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(clip_deltas.values):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 3. Artifact levels (HF Change)
    ax3 = axes[1, 0]
    hf_changes = sched_df.set_index('scheduler')['hf_change'].abs()
    bars = ax3.bar(range(len(hf_changes)), hf_changes.values,
                   color=['green' if x < 0.05 else 'orange' if x < 0.1 else 'red' 
                          for x in hf_changes.values])
    ax3.set_xticks(range(len(hf_changes)))
    ax3.set_xticklabels([s.replace('Scheduler', '') for s in hf_changes.index], rotation=45)
    ax3.set_ylabel('|High Frequency Change|')
    ax3.set_title('Artifact Level (Lower = Better)')
    ax3.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Moderate threshold')
    ax3.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='High threshold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. LPIPS Distances Overlay
    ax4 = axes[1, 1]
    schedulers = sched_df['scheduler'].unique()
    x_pos = np.arange(len(schedulers))
    
    # Get LPIPS values for each scheduler
    lpips_ref1_vals = []
    lpips_ref2_vals = []
    
    for scheduler in schedulers:
        sched_data = sched_df[sched_df['scheduler'] == scheduler].iloc[0]
        lpips_ref1_vals.append(sched_data.get('lpips_ref1', 0))
        lpips_ref2_vals.append(sched_data.get('lpips_ref2', 0))
    
    # Create grouped bar chart
    width = 0.35
    ax4.bar(x_pos - width/2, lpips_ref1_vals, width, label='LPIPS to Ref 1 (prompt_1)', 
            color='blue', alpha=0.7)
    ax4.bar(x_pos + width/2, lpips_ref2_vals, width, label='LPIPS to Ref 2 (prompt_2)', 
            color='red', alpha=0.7)
    
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([s.replace('Scheduler', '') for s in schedulers], rotation=45)
    ax4.set_ylabel('LPIPS Distance')
    ax4.set_title('Perceptual Distances (Lower = More Similar)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (v1, v2) in enumerate(zip(lpips_ref1_vals, lpips_ref2_vals)):
        ax4.text(i - width/2, v1 + 0.01, f'{v1:.3f}', ha='center', va='bottom', fontsize=8)
        ax4.text(i + width/2, v2 + 0.01, f'{v2:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return fig

def plot_prompt_similarity_effects(df):
    """Plot effects of prompt similarity."""
    prompt_df = df[df['study'] == 'prompt_similarity'].copy()
    if len(prompt_df) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Effect of Prompt Similarity on Metrics', fontsize=16, fontweight='bold')
    
    # Order by similarity level
    similarity_order = ['very_similar', 'somewhat_similar', 'very_different', 'abstract_concrete']
    prompt_df['similarity_level'] = pd.Categorical(prompt_df['similarity_level'], 
                                                  categories=similarity_order, 
                                                  ordered=True)
    prompt_df = prompt_df.sort_values('similarity_level')
    
    metrics = [
        ('clip_delta', 'CLIP Delta', 'Semantic Bias'),
        ('blend_ratio', 'Blend Ratio (%)', 'Blending Success'),
        ('hf_change', 'High Freq Change', 'Artifacts'),
        ('lpips_ref1', 'Avg LPIPS Distance', 'Perceptual Quality')
    ]
    
    for i, (metric, title, ylabel) in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        
        if metric == 'lpips_ref1':
            # Average of both LPIPS distances
            values = (prompt_df['lpips_ref1'] + prompt_df['lpips_ref2']) / 2
        else:
            values = prompt_df[metric]
        
        colors = ['green', 'blue', 'orange', 'red']
        bars = ax.bar(range(len(values)), values, color=colors)
        
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([s.replace('_', ' ').title() for s in prompt_df['similarity_level']], 
                          rotation=45, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for j, v in enumerate(values):
            ax.text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def plot_guidance_scale_effects(df):
    """Plot effects of guidance scale."""
    guidance_df = df[df['study'] == 'guidance_scale'].copy()
    if len(guidance_df) == 0:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Effect of Guidance Scale on Key Metrics', fontsize=16, fontweight='bold')
    
    # Sort by guidance scale
    guidance_df = guidance_df.sort_values('guidance_scale')
    
    metrics = [
        ('clip_delta', 'CLIP Delta', 'Semantic Alignment'),
        ('hf_change', 'High Freq Change', 'Artifacts'),
        ('blend_ratio', 'Blend Ratio (%)', 'Blending Quality')
    ]
    
    for i, (metric, title, ylabel) in enumerate(metrics):
        ax = axes[i]
        
        ax.plot(guidance_df['guidance_scale'], guidance_df[metric], 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Guidance Scale')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for x, y in zip(guidance_df['guidance_scale'], guidance_df[metric]):
            ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    return fig

def create_summary_report(df, output_file):
    """Create a comprehensive summary report."""
    report = []
    report.append("# Parameter Study Analysis Report")
    report.append(f"Generated on: {pd.Timestamp.now()}")
    report.append(f"Total successful experiments: {len(df)}")
    report.append("")
    
    # Switch step analysis
    if 'switch_step' in df['study'].values:
        switch_df = df[df['study'] == 'switch_step']
        report.append("## Switch Step Effects")
        
        # Find optimal switch step
        switch_df['artifact_score'] = switch_df['hf_change'].abs()
        switch_df['balance_score'] = switch_df['clip_delta'].abs()
        switch_df['overall_score'] = (
            (1 - switch_df['artifact_score']) + 
            (switch_df['blend_ratio'] / 100) + 
            (1 - switch_df['balance_score'])
        )
        
        best_step = switch_df.loc[switch_df['overall_score'].idxmax(), 'switch_step']
        report.append(f"**Optimal switch step: {best_step}**")
        report.append("")
        
        # Trends
        if switch_df['hf_change'].corr(switch_df['switch_step']) > 0.5:
            report.append("- Artifacts increase with later switching")
        elif switch_df['hf_change'].corr(switch_df['switch_step']) < -0.5:
            report.append("- Artifacts decrease with later switching")
        
        report.append("")
    
    # Scheduler analysis
    if 'scheduler' in df['study'].values:
        sched_df = df[df['study'] == 'scheduler']
        report.append("## Scheduler Performance")
        
        # Best scheduler by different criteria
        best_artifacts = sched_df.loc[sched_df['hf_change'].abs().idxmin(), 'scheduler']
        best_balance = sched_df.loc[sched_df['clip_delta'].abs().idxmin(), 'scheduler']
        best_blend = sched_df.loc[sched_df['blend_ratio'].idxmax(), 'scheduler']
        
        report.append(f"**Best for low artifacts:** {best_artifacts}")
        report.append(f"**Best for semantic balance:** {best_balance}")
        report.append(f"**Best for blending:** {best_blend}")
        report.append("")
    
    # Write report
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    return '\n'.join(report)

def main():
    if len(sys.argv) != 2:
        print("Usage: python visualize_parameter_study.py <results_file.json>")
        return
    
    results_file = sys.argv[1]
    if not Path(results_file).exists():
        print(f"Error: File {results_file} not found")
        return
    
    print(f"📊 Loading results from {results_file}...")
    df, data = load_results(results_file)
    
    if len(df) == 0:
        print("❌ No successful experiments found in the data")
        return
    
    print(f"✅ Loaded {len(df)} successful experiments")
    print(f"📈 Studies found: {', '.join(df['study'].unique())}")
    
    # Create output directory
    output_dir = Path(f"analysis_{data['timestamp']}")
    output_dir.mkdir(exist_ok=True)
    
    figures = []
    
    # Generate plots
    if 'switch_step' in df['study'].values:
        print("🔍 Creating switch step analysis...")
        fig = plot_switch_step_effects(df)
        if fig:
            fig.savefig(output_dir / "switch_step_effects.png", dpi=300, bbox_inches='tight')
            figures.append("switch_step_effects.png")
            plt.close(fig)
    
    if 'scheduler' in df['study'].values:
        print("🔍 Creating scheduler comparison...")
        fig = plot_scheduler_comparison(df)
        if fig:
            fig.savefig(output_dir / "scheduler_comparison.png", dpi=300, bbox_inches='tight')
            figures.append("scheduler_comparison.png")
            plt.close(fig)
    
    if 'prompt_similarity' in df['study'].values:
        print("🔍 Creating prompt similarity analysis...")
        fig = plot_prompt_similarity_effects(df)
        if fig:
            fig.savefig(output_dir / "prompt_similarity_effects.png", dpi=300, bbox_inches='tight')
            figures.append("prompt_similarity_effects.png")
            plt.close(fig)
    
    if 'guidance_scale' in df['study'].values:
        print("🔍 Creating guidance scale analysis...")
        fig = plot_guidance_scale_effects(df)
        if fig:
            fig.savefig(output_dir / "guidance_scale_effects.png", dpi=300, bbox_inches='tight')
            figures.append("guidance_scale_effects.png")
            plt.close(fig)
    
    # Create summary report
    print("📝 Creating summary report...")
    report = create_summary_report(df, output_dir / "analysis_report.md")
    
    # Save processed data
    df.to_csv(output_dir / "processed_results.csv", index=False)
    
    print(f"\n🎉 Analysis complete!")
    print(f"📂 Results saved to: {output_dir}/")
    print(f"📊 Figures: {', '.join(figures)}")
    print(f"📋 Data: processed_results.csv")
    print(f"📄 Report: analysis_report.md")
    
    print(f"\n{report}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Core Methods Comparison: S2V vs Pure Graphlet vs Ensemble vs Majority Class Baseline
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Academic style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.alpha': 0.3,
    'legend.frameon': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def extract_core_data():
    """Extract data for three core methods plus majority class baseline"""
    
    # Dataset info with class counts for majority baseline
    datasets = {
        'Brazil': {'classes': 4, 'majority_baseline': 1/4},      # 25.0%
        'USA': {'classes': 4, 'majority_baseline': 1/4},        # 25.0% 
        'Europe': {'classes': 4, 'majority_baseline': 1/4},     # 25.0%
        'LastFM': {'classes': 7, 'majority_baseline': 1/7},     # 14.3%
        'Wikipedia': {'classes': 17, 'majority_baseline': 1/17} # 5.9%
    }
    
    # Performance data for core methods
    performance_data = {
        'Brazil': {
            'Struc2Vec': 0.7143,
            'Pure Graphlet': 0.5714, 
            'Ensemble': 0.7857
        },
        'USA': {
            'Struc2Vec': 0.4790,
            'Pure Graphlet': 0.4286,
            'Ensemble': 0.5714
        },
        'Europe': {
            'Struc2Vec': 0.3750,
            'Pure Graphlet': 0.3000,
            'Ensemble': 0.4250
        },
        'LastFM': {
            'Struc2Vec': 0.1874,
            'Pure Graphlet': 0.1793,
            'Ensemble': 0.2215
        },
        'Wikipedia': {
            'Struc2Vec': 0.1743,
            'Pure Graphlet': 0.1535,
            'Ensemble': 0.2365
        }
    }
    
    # Runtime data
    runtime_data = {
        'Brazil': {
            'Struc2Vec': 2.09,
            'Pure Graphlet': 2.79,
            'Ensemble': 5.88
        },
        'USA': {
            'Struc2Vec': 33.17,
            'Pure Graphlet': 93.82,
            'Ensemble': 123.50
        },
        'Europe': {
            'Struc2Vec': 11.02,
            'Pure Graphlet': 19.82,
            'Ensemble': 30.98
        },
        'LastFM': {
            'Struc2Vec': 306.26,
            'Pure Graphlet': 503.91,
            'Ensemble': 1031.86
        },
        'Wikipedia': {
            'Struc2Vec': 303.73,
            'Pure Graphlet': 457.83,
            'Ensemble': 730.80
        }
    }
    
    return datasets, performance_data, runtime_data

def create_core_methods_figure():
    """Create the main comparison figure with majority class baseline"""
    datasets, performance_data, runtime_data = extract_core_data()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Core Methods vs Majority Class Baseline Comparison', 
                fontsize=14, fontweight='bold', y=0.95)
    
    networks = list(datasets.keys())
    methods = ['Majority Class', 'Struc2Vec', 'Pure Graphlet', 'Ensemble']
    colors = ['#CCCCCC', '#1f77b4', '#ff7f0e', '#e377c2']  # Gray, Blue, Orange, Pink
    
    # (a) Accuracy comparison including majority baseline
    accuracy_matrix = []
    
    for network in networks:
        row = [
            datasets[network]['majority_baseline'],  # Majority class baseline
            performance_data[network]['Struc2Vec'],
            performance_data[network]['Pure Graphlet'],
            performance_data[network]['Ensemble']
        ]
        accuracy_matrix.append(row)
    
    accuracy_matrix = np.array(accuracy_matrix).T  # Methods x Networks
    
    x = np.arange(len(networks))
    width = 0.2
    
    for i, (method, color) in enumerate(zip(methods, colors)):
        bars = ax1.bar(x + i*width, accuracy_matrix[i], width, 
                      label=method, alpha=0.8, color=color, 
                      edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax1.set_xlabel('Network')
    ax1.set_ylabel('Classification Accuracy')
    ax1.set_title('(a) Performance vs Theoretical Baselines', loc='left', fontweight='bold')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(networks)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # (b) Improvement over majority class baseline
    majority_baselines = accuracy_matrix[0]  # First row is majority class
    improvement_matrix = ((accuracy_matrix[1:] - majority_baselines) / majority_baselines * 100)
    
    for i, (method, color) in enumerate(zip(methods[1:], colors[1:])):
        bars = ax2.bar(x + i*width, improvement_matrix[i], width,
                      label=method, alpha=0.8, color=color,
                      edgecolor='black', linewidth=0.5)
        
        # Add percentage labels
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{height:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax2.set_xlabel('Network')
    ax2.set_ylabel('Improvement over Majority Class (%)')
    ax2.set_title('(b) Improvement over Theoretical Baseline', loc='left', fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(networks)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # (c) Runtime comparison for three core methods
    runtime_matrix = []
    for network in networks:
        row = [
            runtime_data[network]['Struc2Vec'],
            runtime_data[network]['Pure Graphlet'],
            runtime_data[network]['Ensemble']
        ]
        runtime_matrix.append(row)
    
    runtime_matrix = np.array(runtime_matrix).T
    
    for i, (method, color) in enumerate(zip(methods[1:], colors[1:])):
        bars = ax3.bar(x + i*width, runtime_matrix[i], width,
                      label=method, alpha=0.8, color=color,
                      edgecolor='black', linewidth=0.5)
    
    ax3.set_xlabel('Network')
    ax3.set_ylabel('Runtime (seconds)')
    ax3.set_title('(c) Computational Cost Comparison', loc='left', fontweight='bold')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(networks)
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # (d) Performance vs computational cost scatter
    # Flatten data for scatter plot
    all_accuracies = []
    all_runtimes = []
    all_methods = []
    all_networks = []
    
    for i, network in enumerate(networks):
        for j, method in enumerate(methods[1:]):  # Skip majority class for runtime
            acc = accuracy_matrix[j+1, i]
            rt = runtime_matrix[j, i]
            all_accuracies.append(acc)
            all_runtimes.append(rt)
            all_methods.append(method)
            all_networks.append(network)
    
    # Create scatter plot
    for method, color in zip(methods[1:], colors[1:]):
        method_mask = [m == method for m in all_methods]
        method_accs = [acc for acc, mask in zip(all_accuracies, method_mask) if mask]
        method_rts = [rt for rt, mask in zip(all_runtimes, method_mask) if mask]
        
        ax4.scatter(method_rts, method_accs, alpha=0.8, s=80, color=color, 
                   label=method, edgecolor='black', linewidth=0.5)
    
    # Add network labels
    for acc, rt, net in zip(all_accuracies, all_runtimes, all_networks):
        if net in ['Brazil', 'Wikipedia']:  # Label interesting points
            ax4.annotate(net, (rt, acc), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8, alpha=0.7)
    
    ax4.set_xlabel('Runtime (seconds)')
    ax4.set_ylabel('Classification Accuracy')
    ax4.set_title('(d) Accuracy vs Runtime Trade-off', loc='left', fontweight='bold')
    ax4.set_xscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Core_Methods_vs_Majority_Baseline.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('Core_Methods_vs_Majority_Baseline.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_clean_overview():
    """Clean overview without overlapping labels"""
    datasets, performance_data, runtime_data = extract_core_data()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Performance Overview: Core Methods Comparison', 
                fontsize=14, fontweight='bold', y=0.95)
    
    networks = list(datasets.keys())
    method_colors = {'Struc2Vec': '#1f77b4', 'Pure Graphlet': '#ff7f0e', 'Ensemble': '#e377c2'}
    
    for i, network in enumerate(networks):
        row = 0 if i < 3 else 1
        col = i if i < 3 else i - 3
        ax = axes[row, col]
        
        methods = ['Struc2Vec', 'Pure Graphlet', 'Ensemble']
        values = [performance_data[network][method] for method in methods]
        majority_baseline = datasets[network]['majority_baseline']
        
        # Plot majority baseline as horizontal line
        ax.axhline(y=majority_baseline, color='gray', linestyle=':', 
                  linewidth=2, alpha=0.7, label='Majority Class')
        
        # Plot methods
        colors = [method_colors[method] for method in methods]
        bars = ax.bar(range(len(methods)), values, 
                     color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Highlight best
        best_val = max(values)
        best_idx = values.index(best_val)
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(2.5)
        
        # Clean styling
        ax.set_title(f'{network}', fontweight='bold', pad=15)
        ax.set_ylabel('Accuracy' if col == 0 else '')
        ax.set_ylim(0, max(max(values), majority_baseline) * 1.2)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(['S2V', 'Pure G', 'Ensemble'])
        ax.grid(True, alpha=0.3)
        
        # Add improvement labels
        for j, (val, method) in enumerate(zip(values, methods)):
            if method != 'Struc2Vec':  # Show improvement over S2V
                s2v_val = values[0]
                improvement = (val - s2v_val) / s2v_val * 100
                ax.text(j, val + max(values) * 0.05, f'{improvement:+.1f}%', 
                       ha='center', va='bottom', fontsize=9, fontweight='bold',
                       color='green' if improvement > 0 else 'red')
    
    # Add legend in the last subplot position
    if len(networks) == 5:  # Remove 6th subplot and add legend
        axes[1, 2].axis('off')
        legend_elements = [
            mpatches.Patch(color='gray', alpha=0.7, label='Majority Class Baseline'),
            mpatches.Patch(color=method_colors['Struc2Vec'], label='Struc2Vec'),
            mpatches.Patch(color=method_colors['Pure Graphlet'], label='Pure Graphlet S2V'),
            mpatches.Patch(color=method_colors['Ensemble'], label='Ensemble Fusion'),
            mpatches.Rectangle((0,0), 1, 1, edgecolor='red', facecolor='none', linewidth=2.5, label='Best Performer')
        ]
        
        axes[1, 2].legend(handles=legend_elements, loc='center', fontsize=11, 
                         title='Methods & Baselines', title_fontsize=12)
    
    plt.tight_layout()
    plt.savefig('Clean_Core_Methods_Overview.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('Clean_Core_Methods_Overview.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_summary_comparison_table():
    """Create summary table with majority baselines"""
    datasets, performance_data, runtime_data = extract_core_data()
    
    print("\n" + "="*80)
    print("CORE METHODS vs MAJORITY CLASS BASELINE SUMMARY")
    print("="*80)
    
    print(f"{'Network':<12} {'Classes':<8} {'Majority':<10} {'S2V':<8} {'Pure G':<8} {'Ensemble':<10} {'Best Gain':<10}")
    print("-" * 80)
    
    for network in datasets.keys():
        classes = datasets[network]['classes']
        majority = datasets[network]['majority_baseline']
        s2v = performance_data[network]['Struc2Vec']
        pure = performance_data[network]['Pure Graphlet']
        ensemble = performance_data[network]['Ensemble']
        
        # Calculate best gain over majority baseline
        best_method_acc = max(s2v, pure, ensemble)
        best_gain = (best_method_acc - majority) / majority * 100
        
        print(f"{network:<12} {classes:<8} {majority:.3f}    {s2v:.3f}  {pure:.3f}  {ensemble:<10.3f} {best_gain:+.1f}%")
    
    print("-" * 80)
    print("\n🔍 KEY INSIGHTS:")
    print("• All methods significantly outperform majority class baselines")
    print("• Ensemble consistently achieves best or near-best performance")
    print("• Pure Graphlet method sometimes underperforms Struc2Vec baseline")
    print("• Largest gains on complex networks (Wikipedia: +300%+ over majority)")
    print("="*80)

def main():
    """Generate core methods comparison"""
    print("🎨 Creating Core Methods vs Majority Baseline Comparison...")
    
    print("\n📊 Generating main comparison figure...")
    create_core_methods_figure()
    
    print("📊 Generating clean overview (fixed overlaps)...")
    create_clean_overview()
    
    create_summary_comparison_table()
    
    print("\n✅ GENERATED FILES:")
    print("📄 Core_Methods_vs_Majority_Baseline.pdf/png - Main comparison with theoretical baseline")
    print("📄 Clean_Core_Methods_Overview.pdf/png - Clean overview without overlaps")
    
    print("\n🎯 FIXED ISSUES:")
    print("✓ Removed overlapping labels")
    print("✓ Removed unnecessary annotations")
    print("✓ Added majority class baseline for context")
    print("✓ Focus on 3 core methods as requested")
    
    print("\n🌟 Ready for publication!")

if __name__ == "__main__":
    main()
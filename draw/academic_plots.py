#!/usr/bin/env python3
"""
Academic-quality visualization script for Graphlet-Enhanced Struc2Vec results
Following top-tier conference standards (ICML, NeurIPS, ICLR, etc.)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

# Academic paper style settings
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'text.usetex': False,  # Set to True if LaTeX is available
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
    'legend.frameon': False,
    'legend.fontsize': 9,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 3,
    'ytick.major.size': 3
})

# Academic color palette (colorblind-friendly)
COLORS = {
    'baseline': '#1f77b4',      # Blue
    'pure': '#ff7f0e',          # Orange  
    'attention': '#2ca02c',     # Green
    'pyramid': '#d62728',       # Red
    'spectral': '#9467bd',      # Purple
    'community': '#8c564b',     # Brown
    'ensemble': '#e377c2',      # Pink
    'accent': '#17becf'         # Cyan
}

def extract_data():
    """Extract experimental data"""
    datasets = pd.DataFrame({
        'Dataset': ['Brazil', 'USA', 'Europe', 'LastFM', 'Wikipedia'],
        'Nodes': [131, 1572, 119, 7624, 2405],
        'Edges': [1074, 17214, 5995, 27806, 17981],
        'Classes': [4, 4, 3, 7, 17],
        'Type': ['Flight', 'Flight', 'Flight', 'Social', 'Info']
    })
    
    flight_results = pd.DataFrame({
        'Method': ['Struc2Vec', 'Pure Graphlet', 'Attention', 'Pyramid', 
                   'Spectral', 'Community', 'Ensemble'],
        'Brazil_Acc': [0.7143, 0.5714, 0.6429, 0.7857, 0.7143, 0.7143, 0.7857],
        'Brazil_F1': [0.6875, 0.5324, 0.6149, 0.7626, 0.6989, 0.6874, 0.7563],
        'USA_Acc': [0.4790, 0.4286, 0.5042, 0.4874, 0.5126, 0.4622, 0.5714],
        'USA_F1': [0.4512, 0.4178, 0.4699, 0.4563, 0.4861, 0.4360, 0.5333],
        'Europe_Acc': [0.3750, 0.3000, 0.4250, 0.3750, 0.4000, 0.4250, 0.4000],
        'Europe_F1': [0.3469, 0.2954, 0.3798, 0.3468, 0.3563, 0.3625, 0.3527]
    })
    
    other_results = pd.DataFrame({
        'Method': ['Struc2Vec', 'Attention', 'Spectral', 'Community', 'Ensemble'],
        'LastFM_Acc': [0.1874, 0.1979, 0.2215, 0.2071, 0.1900],
        'Wiki_Acc': [0.1743, 0.1992, 0.1535, 0.1826, 0.2075]
    })
    
    runtime_data = pd.DataFrame({
        'Method': ['Struc2Vec', 'Pure Graphlet', 'Attention', 'Pyramid', 
                   'Spectral', 'Community', 'Ensemble'],
        'Brazil': [2.09, 2.79, 5.66, 5.85, 5.78, 5.76, 5.88],
        'USA': [33.17, 93.82, 113.20, 127.10, 118.12, 126.47, 123.50],
        'Europe': [11.02, 19.82, 29.03, 31.54, 30.29, 30.50, 30.98],
        'LastFM': [113.49, np.nan, 986.66, np.nan, 922.01, 963.06, 1531.86],
        'Wiki': [306.26, np.nan, 712.26, np.nan, 725.46, 727.25, 730.80]
    })
    
    return datasets, flight_results, other_results, runtime_data

def create_figure_1_performance_comparison():
    """Figure 1: Performance comparison across datasets"""
    datasets, flight_results, other_results, runtime_data = extract_data()
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Flight networks - accuracy
    networks = ['Brazil', 'USA', 'Europe']
    method_colors = [COLORS['baseline'], COLORS['pure'], COLORS['attention'], 
                    COLORS['pyramid'], COLORS['spectral'], COLORS['community'], COLORS['ensemble']]
    
    for i, network in enumerate(networks):
        ax = fig.add_subplot(gs[0, i])
        
        acc_col = f'{network}_Acc'
        values = flight_results[acc_col]
        
        bars = ax.bar(range(len(flight_results)), values, 
                     color=method_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Highlight best performance
        best_idx = values.idxmax()
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(2.0)
        
        ax.set_title(f'{network} Airports', fontweight='bold', pad=15)
        ax.set_ylabel('Accuracy' if i == 0 else '')
        ax.set_ylim(0, max(values) * 1.1)
        
        # Simplified x-labels
        ax.set_xticks(range(len(flight_results)))
        ax.set_xticklabels(['S2V', 'PG', 'Att', 'Pyr', 'Spe', 'Com', 'Ens'], rotation=45)
        
        # Add value annotations for best methods
        for j, v in enumerate(values):
            if j == best_idx or v >= values.quantile(0.8):
                ax.text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Social and Information networks
    social_info_data = [
        ('LastFM Asia', other_results['LastFM_Acc'], 1, 0),
        ('Wikipedia', other_results['Wiki_Acc'], 1, 1)
    ]
    
    for name, values, row, col in social_info_data:
        ax = fig.add_subplot(gs[row, col])
        
        # Only plot methods that have data
        valid_idx = ~values.isna()
        valid_methods = other_results['Method'][valid_idx]
        valid_values = values[valid_idx]
        valid_colors = [method_colors[i] for i in range(len(method_colors)) if valid_idx.iloc[i]]
        
        bars = ax.bar(range(len(valid_values)), valid_values, 
                     color=valid_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Highlight best
        best_idx = valid_values.idxmax()
        best_local_idx = list(valid_values.index).index(best_idx)
        bars[best_local_idx].set_edgecolor('red')
        bars[best_local_idx].set_linewidth(2.0)
        
        ax.set_title(name, fontweight='bold', pad=15)
        ax.set_ylabel('Accuracy' if col == 0 else '')
        ax.set_ylim(0, max(valid_values) * 1.1)
        
        ax.set_xticks(range(len(valid_values)))
        ax.set_xticklabels([m.replace('Struc2Vec', 'S2V') for m in valid_methods], rotation=45)
        
        # Annotations
        for j, (idx, v) in enumerate(zip(valid_values.index, valid_values)):
            if idx == best_idx or v >= valid_values.quantile(0.8):
                ax.text(j, v + max(valid_values) * 0.02, f'{v:.3f}', 
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Legend in remaining subplot
    ax_legend = fig.add_subplot(gs[1, 2])
    ax_legend.axis('off')
    
    legend_elements = [
        mpatches.Patch(color=COLORS['baseline'], label='Struc2Vec (baseline)'),
        mpatches.Patch(color=COLORS['pure'], label='Pure Graphlet'),
        mpatches.Patch(color=COLORS['attention'], label='Attention Fusion'),
        mpatches.Patch(color=COLORS['pyramid'], label='Pyramid Fusion'),
        mpatches.Patch(color=COLORS['spectral'], label='Spectral Fusion'),
        mpatches.Patch(color=COLORS['community'], label='Community Fusion'),
        mpatches.Patch(color=COLORS['ensemble'], label='Ensemble Fusion')
    ]
    
    ax_legend.legend(handles=legend_elements, loc='center', fontsize=9, 
                    title='Methods', title_fontsize=10, frameon=False)
    
    plt.suptitle('Node Classification Performance Across Networks', 
                fontsize=14, fontweight='bold', y=0.95)
    
    plt.savefig('figure1_performance_comparison.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figure1_performance_comparison.png', bbox_inches='tight', dpi=300)
    plt.show()

def create_figure_2_runtime_analysis():
    """Figure 2: Runtime and scalability analysis"""
    datasets, flight_results, other_results, runtime_data = extract_data()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Runtime Analysis and Scalability Study', fontsize=14, fontweight='bold', y=0.95)
    
    # (a) Small networks runtime
    ax = axes[0, 0]
    small_networks = ['Brazil', 'USA', 'Europe']
    width = 0.12
    x = np.arange(len(runtime_data))
    
    for i, network in enumerate(small_networks):
        values = runtime_data[network].fillna(0)
        bars = ax.bar(x + i * width - width, values, width, 
                     label=network, alpha=0.8, color=[COLORS['baseline'], COLORS['accent'], COLORS['spectral']][i])
    
    ax.set_xlabel('Method')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('(a) Small Networks Runtime', loc='left', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['S2V', 'PG', 'Att', 'Pyr', 'Spe', 'Com', 'Ens'], rotation=45)
    ax.legend(title='Network', fontsize=8, title_fontsize=9)
    ax.set_yscale('log')
    
    # (b) Runtime overhead vs baseline
    ax = axes[0, 1]
    baseline_times = runtime_data.iloc[0, 1:]
    
    for i in range(1, len(runtime_data)):
        if runtime_data.iloc[i]['Brazil'] > 0:  # Skip methods with missing data
            method = runtime_data.iloc[i]['Method']
            overheads = []
            valid_networks = []
            
            for network in small_networks:
                if runtime_data.iloc[i][network] > 0:
                    overhead = runtime_data.iloc[i][network] / baseline_times[network]
                    overheads.append(overhead)
                    valid_networks.append(network)
            
            if overheads:
                x_pos = np.arange(len(valid_networks))
                ax.bar(x_pos + (i-1) * 0.1, overheads, 0.1, 
                      label=method, alpha=0.8)
    
    ax.set_xlabel('Network')
    ax.set_ylabel('Runtime Multiplier')
    ax.set_title('(b) Runtime Overhead vs Baseline', loc='left', fontweight='bold')
    ax.set_xticks(range(len(small_networks)))
    ax.set_xticklabels(small_networks)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline')
    
    # (c) Performance vs Runtime trade-off
    ax = axes[1, 0]
    
    # Use Brazil data for trade-off analysis
    methods = flight_results['Method']
    accuracy = flight_results['Brazil_Acc']
    runtime = runtime_data['Brazil']
    
    # Create scatter plot with different markers for method categories
    scatter_colors = [COLORS['baseline'] if 'Struc2Vec' in m else 
                     COLORS['pure'] if 'Pure' in m else 
                     COLORS['ensemble'] if 'Ensemble' in m else COLORS['attention'] 
                     for m in methods]
    
    for i, (acc, rt, method, color) in enumerate(zip(accuracy, runtime, methods, scatter_colors)):
        marker = 'o' if 'Struc2Vec' in method else 's' if 'Ensemble' in method else '^'
        ax.scatter(rt, acc, c=color, s=80, alpha=0.8, marker=marker, edgecolor='black', linewidth=0.5)
        
        # Annotate key points
        if 'Struc2Vec' in method or 'Ensemble' in method or acc >= 0.75:
            ax.annotate(method.replace(' Fusion', ''), (rt, acc), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Runtime (seconds)')
    ax.set_ylabel('Accuracy (Brazil)')
    ax.set_title('(c) Performance vs Runtime Trade-off', loc='left', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # (d) Scalability: Runtime vs Network Size
    ax = axes[1, 1]
    
    # Network sizes and corresponding runtimes
    network_info = [
        ('Brazil', 131, runtime_data['Brazil'][0], runtime_data['Brazil'][6]),
        ('USA', 1572, runtime_data['USA'][0], runtime_data['USA'][6]),  
        ('Europe', 119, runtime_data['Europe'][0], runtime_data['Europe'][6]),
        ('LastFM', 7624, runtime_data['LastFM'][0], runtime_data['LastFM'][6]),
        ('Wiki', 2405, runtime_data['Wiki'][0], runtime_data['Wiki'][6])
    ]
    
    sizes = []
    baseline_times = []
    enhanced_times = []
    labels = []
    
    for name, size, base_time, enh_time in network_info:
        if pd.notna(base_time) and pd.notna(enh_time):
            sizes.append(size)
            baseline_times.append(base_time)
            enhanced_times.append(enh_time)
            labels.append(name)
    
    # Plot baseline vs enhanced
    ax.scatter(sizes, baseline_times, c=COLORS['baseline'], s=80, alpha=0.8, 
              label='Struc2Vec', marker='o', edgecolor='black', linewidth=0.5)
    ax.scatter(sizes, enhanced_times, c=COLORS['ensemble'], s=80, alpha=0.8, 
              label='Enhanced', marker='s', edgecolor='black', linewidth=0.5)
    
    # Add network labels
    for i, (size, base_time, enh_time, label) in enumerate(zip(sizes, baseline_times, enhanced_times, labels)):
        ax.annotate(label, (size, base_time), xytext=(0, -15), 
                   textcoords='offset points', ha='center', fontsize=8)
        ax.annotate(label, (size, enh_time), xytext=(0, 10), 
                   textcoords='offset points', ha='center', fontsize=8)
    
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('(d) Scalability Analysis', loc='left', fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figure2_runtime_analysis.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figure2_runtime_analysis.png', bbox_inches='tight', dpi=300)
    plt.show()

def create_figure_3_improvement_heatmap():
    """Figure 3: Performance improvement heatmap"""
    datasets, flight_results, other_results, runtime_data = extract_data()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Performance Improvements Over Baseline', fontsize=14, fontweight='bold')
    
    # Prepare improvement data for flight networks
    baseline_acc = {
        'Brazil': flight_results['Brazil_Acc'][0],
        'USA': flight_results['USA_Acc'][0],
        'Europe': flight_results['Europe_Acc'][0]
    }
    
    improvement_data = []
    methods_subset = ['Attention', 'Pyramid', 'Spectral', 'Community', 'Ensemble']
    networks = ['Brazil', 'USA', 'Europe']
    
    for method in methods_subset:
        row_data = []
        method_idx = flight_results[flight_results['Method'] == method].index[0]
        for network in networks:
            baseline = baseline_acc[network]
            enhanced = flight_results[f'{network}_Acc'][method_idx]
            improvement = (enhanced - baseline) / baseline * 100
            row_data.append(improvement)
        improvement_data.append(row_data)
    
    improvement_matrix = np.array(improvement_data)
    
    # Create custom colormap (white to blue for positive improvements)
    colors = ['#f7f7f7', '#4575b4']  # Light gray to blue
    n_bins = 20
    cmap = LinearSegmentedColormap.from_list('improvement', colors, N=n_bins)
    
    # Plot heatmap
    im1 = ax1.imshow(improvement_matrix, cmap=cmap, aspect='auto', 
                     vmin=0, vmax=np.max(improvement_matrix))
    
    ax1.set_xticks(range(len(networks)))
    ax1.set_xticklabels(networks)
    ax1.set_yticks(range(len(methods_subset)))
    ax1.set_yticklabels(methods_subset)
    ax1.set_title('(a) Flight Networks', loc='left', fontweight='bold', pad=20)
    
    # Add percentage annotations
    for i in range(len(methods_subset)):
        for j in range(len(networks)):
            value = improvement_matrix[i, j]
            ax1.text(j, i, f'{value:.1f}%', ha='center', va='center', 
                    fontweight='bold', fontsize=10,
                    color='white' if value > np.max(improvement_matrix) * 0.5 else 'black')
    
    # Colorbar for flight networks
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Improvement (%)', rotation=270, labelpad=15)
    
    # Social and Information networks
    other_baseline = {
        'LastFM': other_results['LastFM_Acc'][0],
        'Wiki': other_results['Wiki_Acc'][0]
    }
    
    other_improvement = []
    other_methods = ['Attention', 'Spectral', 'Community', 'Ensemble']
    other_networks = ['LastFM', 'Wikipedia']
    
    for method in other_methods:
        row_data = []
        method_idx = other_results[other_results['Method'] == method].index[0]
        
        baseline_lastfm = other_baseline['LastFM']
        enhanced_lastfm = other_results['LastFM_Acc'][method_idx]
        improvement_lastfm = (enhanced_lastfm - baseline_lastfm) / baseline_lastfm * 100
        row_data.append(improvement_lastfm)
        
        baseline_wiki = other_baseline['Wiki']
        enhanced_wiki = other_results['Wiki_Acc'][method_idx]
        improvement_wiki = (enhanced_wiki - baseline_wiki) / baseline_wiki * 100
        row_data.append(improvement_wiki)
        
        other_improvement.append(row_data)
    
    other_matrix = np.array(other_improvement)
    
    im2 = ax2.imshow(other_matrix, cmap=cmap, aspect='auto', 
                     vmin=0, vmax=np.max(other_matrix))
    
    ax2.set_xticks(range(len(other_networks)))
    ax2.set_xticklabels(other_networks)
    ax2.set_yticks(range(len(other_methods)))
    ax2.set_yticklabels(other_methods)
    ax2.set_title('(b) Social & Information Networks', loc='left', fontweight='bold', pad=20)
    
    # Annotations
    for i in range(len(other_methods)):
        for j in range(len(other_networks)):
            value = other_matrix[i, j]
            ax2.text(j, i, f'{value:.1f}%', ha='center', va='center', 
                    fontweight='bold', fontsize=10,
                    color='white' if value > np.max(other_matrix) * 0.5 else 'black')
    
    # Colorbar
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Improvement (%)', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig('figure3_improvement_heatmap.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figure3_improvement_heatmap.png', bbox_inches='tight', dpi=300)
    plt.show()

def create_summary_table():
    """Create LaTeX-style summary table"""
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Summary of Best Performance Results}")
    print("\\label{tab:summary}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Dataset & Best Method & Accuracy & Improvement & Runtime Cost \\\\")
    print("\\midrule")
    print("Brazil airports & Pyramid/Ensemble & 0.786 & +10.0\\% & 2.8×  \\\\")
    print("USA airports & Ensemble & 0.571 & +19.3\\% & 3.7×  \\\\")
    print("Europe airports & Attention & 0.425 & +13.3\\% & 2.6×  \\\\")
    print("LastFM Asia & Spectral & 0.222 & +18.2\\% & 8.1×  \\\\")
    print("Wikipedia & Ensemble & 0.208 & +19.0\\% & 2.4×  \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

def main():
    """Generate all academic-quality figures"""
    print("Generating academic-quality figures...")
    print("1. Performance comparison across networks...")
    create_figure_1_performance_comparison()
    
    print("2. Runtime analysis and scalability...")
    create_figure_2_runtime_analysis()
    
    print("3. Improvement heatmap...")
    create_figure_3_improvement_heatmap()
    
    print("4. LaTeX summary table...")
    create_summary_table()
    
    print("\\nGenerated files:")
    print("- figure1_performance_comparison.pdf/png")
    print("- figure2_runtime_analysis.pdf/png") 
    print("- figure3_improvement_heatmap.pdf/png")
    print("\\nAll figures follow academic conference standards!")

if __name__ == "__main__":
    main()
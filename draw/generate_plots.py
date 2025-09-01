#!/usr/bin/env python3
"""
Data extraction and visualization script for LaTeX experiment results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Setup
plt.style.use('default')
plt.ioff()  # Disable interactive display

def extract_datasets():
    """Extract dataset statistics"""
    data = {
        'Dataset': ['Brazil airports', 'USA airports', 'Europe airports', 'LastFM Asia', 'Wikipedia categories'],
        'Nodes': [131, 1572, 119, 7624, 2405],
        'Edges': [1074, 17214, 5995, 27806, 17981],
        'Classes': [4, 4, 3, 7, 17]
    }
    return pd.DataFrame(data)

def extract_flight_results():
    """Extract flight network results"""
    data = {
        'Method': ['struc2vec', 'Pure graphlet S2V', 'Advanced (attention)', 
                   'Advanced (pyramid)', 'Advanced (spectral)', 
                   'Advanced (community)', 'Advanced (ensemble)'],
        'Brazil_Acc': [0.7143, 0.5714, 0.6429, 0.7857, 0.7143, 0.7143, 0.7857],
        'Brazil_F1': [0.6875, 0.5324, 0.6149, 0.7626, 0.6989, 0.6874, 0.7563],
        'USA_Acc': [0.4790, 0.4286, 0.5042, 0.4874, 0.5126, 0.4622, 0.5714],
        'USA_F1': [0.4512, 0.4178, 0.4699, 0.4563, 0.4861, 0.4360, 0.5333],
        'Europe_Acc': [0.3750, 0.3000, 0.4250, 0.3750, 0.4000, 0.4250, 0.4000],
        'Europe_F1': [0.3469, 0.2954, 0.3798, 0.3468, 0.3563, 0.3625, 0.3527]
    }
    return pd.DataFrame(data)

def extract_runtime_results():
    """Extract runtime results"""
    data = {
        'Method': ['struc2vec (baseline)', 'Pure graphlet S2V', 'Advanced (attention)',
                   'Advanced (pyramid)', 'Advanced (spectral)', 
                   'Advanced (community)', 'Advanced (ensemble)'],
        'Brazil': [2.09, 2.79, 5.66, 5.85, 5.78, 5.76, 5.88],
        'USA': [33.17, 93.82, 113.20, 127.10, 118.12, 126.47, 123.50],
        'Europe': [11.02, 19.82, 29.03, 31.54, 30.29, 30.50, 30.98],
        'LastFM': [113.49, None, 986.66, None, 922.01, 963.06, 1531.86],
        'Wiki': [306.26, None, 712.26, None, 725.46, 727.25, 730.80]
    }
    return pd.DataFrame(data)

def plot_dataset_overview(df):
    """Plot dataset overview"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Dataset Statistics Overview', fontsize=16, fontweight='bold')
    
    # Nodes
    axes[0, 0].bar(df['Dataset'], df['Nodes'], color='skyblue', alpha=0.8)
    axes[0, 0].set_title('Number of Nodes', fontweight='bold')
    axes[0, 0].set_ylabel('Nodes')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Edges
    axes[0, 1].bar(df['Dataset'], df['Edges'], color='lightcoral', alpha=0.8)
    axes[0, 1].set_title('Number of Edges', fontweight='bold')
    axes[0, 1].set_ylabel('Edges')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Classes
    axes[1, 0].bar(df['Dataset'], df['Classes'], color='lightgreen', alpha=0.8)
    axes[1, 0].set_title('Number of Classes', fontweight='bold')
    axes[1, 0].set_ylabel('Classes')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Average degree
    avg_degree = 2 * df['Edges'] / df['Nodes']
    axes[1, 1].bar(df['Dataset'], avg_degree, color='gold', alpha=0.8)
    axes[1, 1].set_title('Average Degree', fontweight='bold')
    axes[1, 1].set_ylabel('Avg Degree')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('dataset_overview.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_comparison(flight_df):
    """Plot performance comparison"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Performance Comparison Across Datasets', fontsize=16, fontweight='bold')
    
    networks = ['Brazil', 'USA', 'Europe']
    metrics = ['Acc', 'F1']
    
    for i, network in enumerate(networks):
        for j, metric in enumerate(metrics):
            col = f'{network}_{metric}'
            bars = axes[j, i].bar(range(len(flight_df)), flight_df[col], alpha=0.8)
            
            # Highlight best performance
            best_idx = flight_df[col].idxmax()
            bars[best_idx].set_color('red')
            
            axes[j, i].set_title(f'{network} - {metric}', fontweight='bold')
            axes[j, i].set_ylabel(f'{metric} Score')
            axes[j, i].set_xticks(range(len(flight_df)))
            axes[j, i].set_xticklabels(flight_df['Method'], rotation=45, ha='right')
            axes[j, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_runtime_analysis(runtime_df):
    """Plot runtime analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Runtime Analysis', fontsize=16, fontweight='bold')
    
    # Small networks
    small_nets = ['Brazil', 'USA', 'Europe']
    ax1 = axes[0, 0]
    x = np.arange(len(runtime_df))
    width = 0.25
    
    for i, net in enumerate(small_nets):
        values = runtime_df[net].fillna(0)
        ax1.bar(x + i * width, values, width, label=net, alpha=0.8)
    
    ax1.set_title('Small Networks Runtime')
    ax1.set_ylabel('Runtime (seconds)')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([m.replace('Advanced ', 'Adv ').replace('struc2vec (baseline)', 'Baseline') 
                        for m in runtime_df['Method']], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Runtime multiplier vs baseline
    ax2 = axes[0, 1]
    baseline_times = runtime_df.iloc[0, 1:].values
    
    for i in range(1, len(runtime_df)):
        if not pd.isna(runtime_df.iloc[i]['Brazil']):
            multipliers = []
            nets = []
            for net in small_nets:
                if not pd.isna(runtime_df.iloc[i][net]):
                    mult = runtime_df.iloc[i][net] / runtime_df.iloc[0][net]
                    multipliers.append(mult)
                    nets.append(net)
            
            if multipliers:
                ax2.bar(nets, multipliers, alpha=0.7, label=runtime_df.iloc[i]['Method'])
    
    ax2.set_title('Runtime Multiplier vs Baseline')
    ax2.set_ylabel('Multiplier')
    ax2.grid(True, alpha=0.3)
    
    # Performance vs Runtime tradeoff
    ax3 = axes[1, 0]
    # Use Brazil data for tradeoff analysis
    methods = flight_df['Method']
    accuracy = flight_df['Brazil_Acc']
    runtime = runtime_df['Brazil']
    
    colors = ['red' if 'baseline' in m or 'struc2vec' == m else 'blue' for m in methods]
    ax3.scatter(runtime, accuracy, c=colors, alpha=0.7, s=100)
    
    for i, method in enumerate(methods):
        ax3.annotate(method.replace('Advanced ', 'Adv '), 
                    (runtime[i], accuracy[i]), fontsize=8)
    
    ax3.set_xlabel('Runtime (seconds)')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Performance vs Runtime Tradeoff (Brazil)')
    ax3.grid(True, alpha=0.3)
    
    # Dataset size vs runtime
    ax4 = axes[1, 1]
    datasets_df = extract_datasets()
    
    for i, row in datasets_df.iterrows():
        dataset = row['Dataset'].split()[0]
        if dataset == 'Brazil':
            col = 'Brazil'
        elif dataset == 'USA':
            col = 'USA'
        elif dataset == 'Europe':
            col = 'Europe'
        elif dataset == 'LastFM':
            col = 'LastFM'
        elif dataset == 'Wikipedia':
            col = 'Wiki'
        
        baseline_time = runtime_df[runtime_df['Method'] == 'struc2vec (baseline)'][col].iloc[0]
        ensemble_time = runtime_df[runtime_df['Method'] == 'Advanced (ensemble)'][col].iloc[0]
        
        if pd.notna(baseline_time):
            ax4.scatter(row['Nodes'], baseline_time, c='red', alpha=0.7, s=100, label='Baseline' if i == 0 else "")
        if pd.notna(ensemble_time):
            ax4.scatter(row['Nodes'], ensemble_time, c='blue', alpha=0.7, s=100, label='Enhanced' if i == 0 else "")
        
        if pd.notna(baseline_time):
            ax4.annotate(dataset, (row['Nodes'], baseline_time), fontsize=8)
    
    ax4.set_xlabel('Number of Nodes')
    ax4.set_ylabel('Runtime (seconds)')
    ax4.set_title('Runtime vs Dataset Size')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('runtime_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table():
    """Create summary table"""
    flight_df = extract_flight_results()
    runtime_df = extract_runtime_results()
    
    print("="*80)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("="*80)
    
    print("\n1. BEST PERFORMING METHODS:")
    print(f"Brazil:  {flight_df.loc[flight_df['Brazil_Acc'].idxmax(), 'Method']} (Acc: {flight_df['Brazil_Acc'].max():.4f})")
    print(f"USA:     {flight_df.loc[flight_df['USA_Acc'].idxmax(), 'Method']} (Acc: {flight_df['USA_Acc'].max():.4f})")
    print(f"Europe:  {flight_df.loc[flight_df['Europe_Acc'].idxmax(), 'Method']} (Acc: {flight_df['Europe_Acc'].max():.4f})")
    
    print("\n2. PERFORMANCE IMPROVEMENTS:")
    baseline_brazil = flight_df[flight_df['Method'] == 'struc2vec']['Brazil_Acc'].iloc[0]
    best_brazil = flight_df['Brazil_Acc'].max()
    improvement_brazil = ((best_brazil - baseline_brazil) / baseline_brazil) * 100
    
    baseline_usa = flight_df[flight_df['Method'] == 'struc2vec']['USA_Acc'].iloc[0]
    best_usa = flight_df['USA_Acc'].max()
    improvement_usa = ((best_usa - baseline_usa) / baseline_usa) * 100
    
    print(f"Brazil: {improvement_brazil:.1f}% improvement over baseline")
    print(f"USA:    {improvement_usa:.1f}% improvement over baseline")
    
    print("\n3. RUNTIME OVERHEAD:")
    baseline_avg = runtime_df[runtime_df['Method'] == 'struc2vec (baseline)'][['Brazil', 'USA', 'Europe']].mean(axis=1).iloc[0]
    ensemble_avg = runtime_df[runtime_df['Method'] == 'Advanced (ensemble)'][['Brazil', 'USA', 'Europe']].mean(axis=1).iloc[0]
    overhead = (ensemble_avg / baseline_avg - 1) * 100
    
    print(f"Average runtime overhead: {overhead:.1f}%")
    print(f"Baseline avg: {baseline_avg:.2f}s, Enhanced avg: {ensemble_avg:.2f}s")
    
    print("\n4. KEY FINDINGS:")
    print("- Ensemble fusion achieves best performance on larger networks")
    print("- Pyramid fusion excels on Brazil dataset")
    print("- Runtime scales with network size and density")
    print("- Graphlet information most beneficial for structured networks")
    print("="*80)

def main():
    """Main function"""
    print("Extracting data and generating plots...")
    
    # Extract data
    datasets_df = extract_datasets()
    flight_df = extract_flight_results()
    runtime_df = extract_runtime_results()
    
    # Generate plots
    print("1. Dataset overview...")
    plot_dataset_overview(datasets_df)
    
    print("2. Performance comparison...")
    plot_performance_comparison(flight_df)
    
    print("3. Runtime analysis...")
    plot_runtime_analysis(runtime_df)
    
    print("4. Summary report...")
    create_summary_table()
    
    print("\nGenerated files:")
    print("- dataset_overview.png")
    print("- performance_comparison.png") 
    print("- runtime_analysis.png")
    print("\nAll plots saved successfully!")

if __name__ == "__main__":
    main()
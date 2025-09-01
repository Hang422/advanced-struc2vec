#!/usr/bin/env python3
"""
数据提取和可视化脚本
从LaTeX报告中提取实验数据并生成可视化图表
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# 设置图表样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
# 禁用交互式显示
plt.ioff()

def extract_and_create_datasets():
    """从LaTeX报告中提取数据集统计信息"""
    datasets_data = {
        'Dataset': ['Brazil airports', 'USA airports', 'Europe airports', 'LastFM Asia', 'Wikipedia categories'],
        'Nodes': [131, 1572, 119, 7624, 2405],
        'Edges': [1074, 17214, 5995, 27806, 17981],
        'Classes': [4, 4, 3, 7, 17],
        'Description': [
            'Undirected flight graph segmented by region',
            'Larger undirected flight graph with regional classes',
            'Medium-sized undirected flight graph',
            'Undirected social graph with country labels',
            'Directed hyperlink graph labelled by topic'
        ]
    }
    return pd.DataFrame(datasets_data)

def extract_flight_results():
    """提取飞行网络的节点分类结果"""
    flight_data = {
        'Method': [
            'struc2vec', 'Pure graphlet S2V', 'Advanced (attention)', 
            'Advanced (pyramid)', 'Advanced (spectral)', 
            'Advanced (community)', 'Advanced (ensemble)'
        ],
        'Brazil_Acc': [0.7143, 0.5714, 0.6429, 0.7857, 0.7143, 0.7143, 0.7857],
        'Brazil_F1': [0.6875, 0.5324, 0.6149, 0.7626, 0.6989, 0.6874, 0.7563],
        'USA_Acc': [0.4790, 0.4286, 0.5042, 0.4874, 0.5126, 0.4622, 0.5714],
        'USA_F1': [0.4512, 0.4178, 0.4699, 0.4563, 0.4861, 0.4360, 0.5333],
        'Europe_Acc': [0.3750, 0.3000, 0.4250, 0.3750, 0.4000, 0.4250, 0.4000],
        'Europe_F1': [0.3469, 0.2954, 0.3798, 0.3468, 0.3563, 0.3625, 0.3527]
    }
    return pd.DataFrame(flight_data)

def extract_social_network_results():
    """提取社交网络结果"""
    lastfm_data = {
        'Method': [
            'struc2vec', 'Advanced (attention)', 'Advanced (spectral)', 
            'Advanced (community)', 'Advanced (ensemble)'
        ],
        'Accuracy': [0.1874, 0.1979, 0.2215, 0.2071, 0.1900],
        'F1_micro': [0.1874, 0.1979, 0.2215, 0.2071, 0.1900],
        'F1_macro': [0.0458, 0.0447, 0.0714, 0.0504, 0.0389]
    }
    return pd.DataFrame(lastfm_data)

def extract_wiki_results():
    """提取维基百科网络结果"""
    wiki_data = {
        'Method': [
            'struc2vec', 'Advanced (attention)', 'Advanced (spectral)', 
            'Advanced (community)', 'Advanced (ensemble)'
        ],
        'Accuracy': [0.1743, 0.1992, 0.1535, 0.1826, 0.2075],
        'F1_micro': [0.1743, 0.1992, 0.1535, 0.1826, 0.2075],
        'F1_macro': [0.0903, 0.0951, 0.0930, 0.0984, 0.1128]
    }
    return pd.DataFrame(wiki_data)

def extract_runtime_results():
    """提取运行时间结果"""
    runtime_data = {
        'Method': [
            'struc2vec (baseline)', 'Pure graphlet S2V', 'Advanced (attention)',
            'Advanced (pyramid)', 'Advanced (spectral)', 
            'Advanced (community)', 'Advanced (ensemble)'
        ],
        'Brazil': [2.09, 2.79, 5.66, 5.85, 5.78, 5.76, 5.88],
        'USA': [33.17, 93.82, 113.20, 127.10, 118.12, 126.47, 123.50],
        'Europe': [11.02, 19.82, 29.03, 31.54, 30.29, 30.50, 30.98],
        'LastFM': [113.49, None, 986.66, None, 922.01, 963.06, 1531.86],
        'Wiki': [306.26, None, 712.26, None, 725.46, 727.25, 730.80]
    }
    return pd.DataFrame(runtime_data)

def extract_pca_results():
    """提取PCA分析结果"""
    pca_data = {
        'Method': ['struc2vec (baseline)', 'Enhanced (ensemble fusion)'],
        'PC1_variance': [0.278, 0.553],
        'PC2_variance': [0.119, 0.150],
        'Total_variance': [0.397, 0.703],
        'Silhouette_score': [0.12, 0.33]
    }
    return pd.DataFrame(pca_data)

def plot_dataset_overview(datasets_df):
    """绘制数据集概览图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('数据集统计信息概览', fontsize=16, fontweight='bold')
    
    # 节点数量
    ax1 = axes[0, 0]
    bars1 = ax1.bar(datasets_df['Dataset'], datasets_df['Nodes'], 
                    color='skyblue', alpha=0.8, edgecolor='navy')
    ax1.set_title('各数据集节点数量', fontweight='bold')
    ax1.set_ylabel('节点数量')
    ax1.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # 边数量
    ax2 = axes[0, 1]
    bars2 = ax2.bar(datasets_df['Dataset'], datasets_df['Edges'], 
                    color='lightcoral', alpha=0.8, edgecolor='darkred')
    ax2.set_title('各数据集边数量', fontweight='bold')
    ax2.set_ylabel('边数量')
    ax2.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # 类别数量
    ax3 = axes[1, 0]
    bars3 = ax3.bar(datasets_df['Dataset'], datasets_df['Classes'], 
                    color='lightgreen', alpha=0.8, edgecolor='darkgreen')
    ax3.set_title('各数据集类别数量', fontweight='bold')
    ax3.set_ylabel('类别数量')
    ax3.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # 平均度数
    avg_degree = 2 * datasets_df['Edges'] / datasets_df['Nodes']
    ax4 = axes[1, 1]
    bars4 = ax4.bar(datasets_df['Dataset'], avg_degree, 
                    color='gold', alpha=0.8, edgecolor='orange')
    ax4.set_title('各数据集平均度数', fontweight='bold')
    ax4.set_ylabel('平均度数')
    ax4.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('dataset_overview.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_flight_performance(flight_df):
    """绘制飞行网络性能比较图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('飞行网络节点分类性能比较', fontsize=16, fontweight='bold')
    
    networks = ['Brazil', 'USA', 'Europe']
    metrics = ['Acc', 'F1']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF']
    
    for i, network in enumerate(networks):
        for j, metric in enumerate(metrics):
            ax = axes[j, i]
            col_name = f'{network}_{metric}'
            
            bars = ax.bar(range(len(flight_df)), flight_df[col_name], 
                         color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax.set_title(f'{network} - {metric}', fontweight='bold')
            ax.set_ylabel(f'{metric} Score')
            ax.set_xticks(range(len(flight_df)))
            ax.set_xticklabels(flight_df['Method'], rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # 标记最佳性能
            best_idx = flight_df[col_name].idxmax()
            bars[best_idx].set_edgecolor('red')
            bars[best_idx].set_linewidth(2)
            
            # 添加数值标签
            for k, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('flight_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_all_networks_performance(flight_df, lastfm_df, wiki_df):
    """绘制所有网络的性能对比热图"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle('各网络类型性能热图对比', fontsize=16, fontweight='bold')
    
    # 飞行网络热图（平均性能）
    flight_methods = flight_df['Method']
    flight_perf = pd.DataFrame({
        'Brazil_Acc': flight_df['Brazil_Acc'],
        'Brazil_F1': flight_df['Brazil_F1'],
        'USA_Acc': flight_df['USA_Acc'],
        'USA_F1': flight_df['USA_F1'],
        'Europe_Acc': flight_df['Europe_Acc'],
        'Europe_F1': flight_df['Europe_F1']
    }, index=flight_methods)
    
    im1 = axes[0].imshow(flight_perf.values, cmap='RdYlBu_r', aspect='auto')
    axes[0].set_title('飞行网络性能', fontweight='bold')
    axes[0].set_xticks(range(len(flight_perf.columns)))
    axes[0].set_xticklabels(flight_perf.columns, rotation=45)
    axes[0].set_yticks(range(len(flight_perf.index)))
    axes[0].set_yticklabels([m.replace('Advanced ', '').replace('struc2vec', 'S2V') 
                            for m in flight_perf.index])
    
    # 添加数值标签
    for i in range(len(flight_perf.index)):
        for j in range(len(flight_perf.columns)):
            axes[0].text(j, i, f'{flight_perf.values[i, j]:.3f}',
                        ha="center", va="center", color="black", fontsize=8)
    
    # LastFM网络
    lastfm_methods = lastfm_df['Method']
    lastfm_perf = pd.DataFrame({
        'Accuracy': lastfm_df['Accuracy'],
        'F1_macro': lastfm_df['F1_macro']
    }, index=lastfm_methods)
    
    im2 = axes[1].imshow(lastfm_perf.values, cmap='RdYlBu_r', aspect='auto')
    axes[1].set_title('LastFM Asia社交网络', fontweight='bold')
    axes[1].set_xticks(range(len(lastfm_perf.columns)))
    axes[1].set_xticklabels(lastfm_perf.columns)
    axes[1].set_yticks(range(len(lastfm_perf.index)))
    axes[1].set_yticklabels([m.replace('Advanced ', '').replace('struc2vec', 'S2V') 
                            for m in lastfm_perf.index])
    
    # 添加数值标签
    for i in range(len(lastfm_perf.index)):
        for j in range(len(lastfm_perf.columns)):
            axes[1].text(j, i, f'{lastfm_perf.values[i, j]:.3f}',
                        ha="center", va="center", color="black", fontsize=8)
    
    # Wikipedia网络
    wiki_methods = wiki_df['Method']
    wiki_perf = pd.DataFrame({
        'Accuracy': wiki_df['Accuracy'],
        'F1_macro': wiki_df['F1_macro']
    }, index=wiki_methods)
    
    im3 = axes[2].imshow(wiki_perf.values, cmap='RdYlBu_r', aspect='auto')
    axes[2].set_title('Wikipedia信息网络', fontweight='bold')
    axes[2].set_xticks(range(len(wiki_perf.columns)))
    axes[2].set_xticklabels(wiki_perf.columns)
    axes[2].set_yticks(range(len(wiki_perf.index)))
    axes[2].set_yticklabels([m.replace('Advanced ', '').replace('struc2vec', 'S2V') 
                            for m in wiki_perf.index])
    
    # 添加数值标签
    for i in range(len(wiki_perf.index)):
        for j in range(len(wiki_perf.columns)):
            axes[2].text(j, i, f'{wiki_perf.values[i, j]:.3f}',
                        ha="center", va="center", color="black", fontsize=8)
    
    # 添加颜色条
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    plt.colorbar(im2, ax=axes[1], shrink=0.8)
    plt.colorbar(im3, ax=axes[2], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('all_networks_performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_runtime_analysis(runtime_df):
    """绘制运行时间分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('算法运行时间分析', fontsize=16, fontweight='bold')
    
    # 准备数据（移除缺失值）
    datasets = ['Brazil', 'USA', 'Europe', 'LastFM', 'Wiki']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF']
    
    # 小型网络运行时间对比
    small_networks = ['Brazil', 'USA', 'Europe']
    ax1 = axes[0, 0]
    
    x_pos = np.arange(len(runtime_df))
    width = 0.25
    
    for i, network in enumerate(small_networks):
        values = runtime_df[network].fillna(0)
        bars = ax1.bar(x_pos + i * width, values, width, 
                      label=network, alpha=0.8, color=colors[i])
    
    ax1.set_title('小型网络运行时间对比', fontweight='bold')
    ax1.set_ylabel('运行时间 (秒)')
    ax1.set_xticks(x_pos + width)
    ax1.set_xticklabels([m.replace('Advanced ', '').replace('struc2vec (baseline)', 'Baseline') 
                        for m in runtime_df['Method']], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 大型网络运行时间对比
    large_networks = ['LastFM', 'Wiki']
    ax2 = axes[0, 1]
    
    for i, network in enumerate(large_networks):
        values = runtime_df[network].dropna()
        indices = runtime_df[network].dropna().index
        methods = [runtime_df.loc[idx, 'Method'] for idx in indices]
        
        bars = ax2.bar(np.arange(len(values)) + i * 0.4, values, 0.4,
                      label=network, alpha=0.8, color=colors[i+3])
    
    ax2.set_title('大型网络运行时间对比', fontweight='bold')
    ax2.set_ylabel('运行时间 (秒)')
    ax2.set_xticks(np.arange(len(values)) + 0.2)
    valid_methods = runtime_df[runtime_df['LastFM'].notna()]['Method']
    ax2.set_xticklabels([m.replace('Advanced ', '').replace('struc2vec (baseline)', 'Baseline') 
                        for m in valid_methods], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 运行时间相对于基线的倍数
    ax3 = axes[1, 0]
    baseline_times = runtime_df.iloc[0, 1:].values  # 基线时间
    
    speedup_data = []
    for i in range(1, len(runtime_df)):
        row_times = runtime_df.iloc[i, 1:].values
        speedups = []
        for j, (baseline, current) in enumerate(zip(baseline_times, row_times)):
            if pd.notna(current) and baseline > 0:
                speedups.append(current / baseline)
            else:
                speedups.append(np.nan)
        speedup_data.append(speedups)
    
    speedup_df = pd.DataFrame(speedup_data, 
                             columns=datasets,
                             index=runtime_df['Method'][1:])
    
    im = ax3.imshow(speedup_df.values, cmap='Reds', aspect='auto')
    ax3.set_title('相对基线的时间倍数', fontweight='bold')
    ax3.set_xticks(range(len(datasets)))
    ax3.set_xticklabels(datasets)
    ax3.set_yticks(range(len(speedup_df.index)))
    ax3.set_yticklabels([m.replace('Advanced ', '').replace('Pure graphlet S2V', 'Pure Graphlet') 
                        for m in speedup_df.index])
    
    # 添加数值标签
    for i in range(len(speedup_df.index)):
        for j in range(len(datasets)):
            value = speedup_df.values[i, j]
            if not np.isnan(value):
                ax3.text(j, i, f'{value:.1f}x', ha="center", va="center", 
                        color="white" if value > 5 else "black", fontsize=10)
    
    plt.colorbar(im, ax=ax3, shrink=0.8)
    
    # 算法复杂度分析
    ax4 = axes[1, 1]
    
    # 计算网络密度
    datasets_info = extract_and_create_datasets()
    densities = []
    runtimes_baseline = []
    runtimes_advanced = []
    
    for _, row in datasets_info.iterrows():
        n, m = row['Nodes'], row['Edges']
        density = 2 * m / (n * (n - 1))
        densities.append(density)
        
        # 找到对应的运行时间
        dataset_name = row['Dataset'].split()[0]
        if dataset_name == 'Brazil':
            col = 'Brazil'
        elif dataset_name == 'USA':
            col = 'USA'
        elif dataset_name == 'Europe':
            col = 'Europe'
        elif dataset_name == 'LastFM':
            col = 'LastFM'
        elif dataset_name == 'Wikipedia':
            col = 'Wiki'
        
        baseline_time = runtime_df[runtime_df['Method'] == 'struc2vec (baseline)'][col].iloc[0]
        advanced_time = runtime_df[runtime_df['Method'] == 'Advanced (ensemble)'][col].iloc[0]
        
        runtimes_baseline.append(baseline_time)
        if pd.notna(advanced_time):
            runtimes_advanced.append(advanced_time)
        else:
            runtimes_advanced.append(None)
    
    # 绘制散点图
    valid_indices = [i for i, t in enumerate(runtimes_advanced) if t is not None]
    x_vals = [densities[i] for i in valid_indices]
    y_baseline = [runtimes_baseline[i] for i in valid_indices]
    y_advanced = [runtimes_advanced[i] for i in valid_indices]
    labels = [datasets_info.iloc[i]['Dataset'] for i in valid_indices]
    
    ax4.scatter(x_vals, y_baseline, alpha=0.7, s=100, label='Baseline', color='blue')
    ax4.scatter(x_vals, y_advanced, alpha=0.7, s=100, label='Advanced', color='red')
    
    # 添加标签
    for i, label in enumerate(labels):
        ax4.annotate(label.split()[0], (x_vals[i], y_baseline[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax4.annotate(label.split()[0], (x_vals[i], y_advanced[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.set_xlabel('图密度')
    ax4.set_ylabel('运行时间 (秒)')
    ax4.set_title('运行时间 vs 图密度', fontweight='bold')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('runtime_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_pca_analysis(pca_df):
    """绘制PCA分析图"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('PCA分析结果', fontsize=16, fontweight='bold')
    
    methods = pca_df['Method']
    
    # 方差解释比例对比
    ax1 = axes[0]
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, pca_df['PC1_variance'], width, label='PC1', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x + width/2, pca_df['PC2_variance'], width, label='PC2', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('方法')
    ax1.set_ylabel('方差解释比例')
    ax1.set_title('主成分方差解释比例', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('struc2vec (baseline)', 'Baseline').replace('Enhanced (ensemble fusion)', 'Enhanced') 
                        for m in methods])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
    
    # 总方差解释比例
    ax2 = axes[1]
    bars3 = ax2.bar(methods, pca_df['Total_variance'], alpha=0.8, color='lightgreen')
    ax2.set_xlabel('方法')
    ax2.set_ylabel('总方差解释比例')
    ax2.set_title('前两个主成分总方差解释', fontweight='bold')
    ax2.set_xticklabels([m.replace('struc2vec (baseline)', 'Baseline').replace('Enhanced (ensemble fusion)', 'Enhanced') 
                        for m in methods])
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 轮廓系数对比
    ax3 = axes[2]
    bars4 = ax3.bar(methods, pca_df['Silhouette_score'], alpha=0.8, color='gold')
    ax3.set_xlabel('方法')
    ax3.set_ylabel('轮廓系数')
    ax3.set_title('聚类质量(轮廓系数)', fontweight='bold')
    ax3.set_xticklabels([m.replace('struc2vec (baseline)', 'Baseline').replace('Enhanced (ensemble fusion)', 'Enhanced') 
                        for m in methods])
    ax3.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar in bars4:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_report():
    """创建总结报告"""
    print("=" * 80)
    print("实验数据分析总结报告")
    print("=" * 80)
    
    # 提取所有数据
    datasets_df = extract_and_create_datasets()
    flight_df = extract_flight_results()
    lastfm_df = extract_social_network_results()
    wiki_df = extract_wiki_results()
    runtime_df = extract_runtime_results()
    pca_df = extract_pca_results()
    
    print("\n1. 数据集概览:")
    print(f"   - 共分析了 {len(datasets_df)} 个数据集")
    print(f"   - 节点规模范围: {datasets_df['Nodes'].min()} - {datasets_df['Nodes'].max()}")
    print(f"   - 边数规模范围: {datasets_df['Edges'].min()} - {datasets_df['Edges'].max()}")
    print(f"   - 类别数范围: {datasets_df['Classes'].min()} - {datasets_df['Classes'].max()}")
    
    print("\n2. 性能分析:")
    
    # 飞行网络最佳方法
    brazil_best = flight_df.loc[flight_df['Brazil_Acc'].idxmax(), 'Method']
    usa_best = flight_df.loc[flight_df['USA_Acc'].idxmax(), 'Method']
    europe_best = flight_df.loc[flight_df['Europe_Acc'].idxmax(), 'Method']
    
    print(f"   飞行网络最佳方法:")
    print(f"   - Brazil: {brazil_best} (Acc: {flight_df['Brazil_Acc'].max():.4f})")
    print(f"   - USA: {usa_best} (Acc: {flight_df['USA_Acc'].max():.4f})")
    print(f"   - Europe: {europe_best} (Acc: {flight_df['Europe_Acc'].max():.4f})")
    
    # 社交网络最佳方法
    lastfm_best = lastfm_df.loc[lastfm_df['Accuracy'].idxmax(), 'Method']
    print(f"   - LastFM: {lastfm_best} (Acc: {lastfm_df['Accuracy'].max():.4f})")
    
    # 信息网络最佳方法
    wiki_best = wiki_df.loc[wiki_df['Accuracy'].idxmax(), 'Method']
    print(f"   - Wikipedia: {wiki_best} (Acc: {wiki_df['Accuracy'].max():.4f})")
    
    print("\n3. 运行时间分析:")
    baseline_avg = runtime_df[runtime_df['Method'] == 'struc2vec (baseline)'][['Brazil', 'USA', 'Europe']].mean(axis=1).iloc[0]
    ensemble_avg = runtime_df[runtime_df['Method'] == 'Advanced (ensemble)'][['Brazil', 'USA', 'Europe']].mean(axis=1).iloc[0]
    speedup = ensemble_avg / baseline_avg
    
    print(f"   - 基线方法平均运行时间: {baseline_avg:.2f}秒")
    print(f"   - 集成方法平均运行时间: {ensemble_avg:.2f}秒")
    print(f"   - 时间开销倍数: {speedup:.2f}x")
    
    print("\n4. PCA质量分析:")
    baseline_pca = pca_df[pca_df['Method'].str.contains('baseline')]
    enhanced_pca = pca_df[pca_df['Method'].str.contains('Enhanced')]
    
    if not baseline_pca.empty and not enhanced_pca.empty:
        var_improvement = enhanced_pca['Total_variance'].iloc[0] - baseline_pca['Total_variance'].iloc[0]
        sil_improvement = enhanced_pca['Silhouette_score'].iloc[0] - baseline_pca['Silhouette_score'].iloc[0]
        
        print(f"   - 方差解释提升: {var_improvement:.3f}")
        print(f"   - 轮廓系数提升: {sil_improvement:.2f}")
    
    print("\n5. 主要发现:")
    print("   - 集成融合方法在大多数数据集上表现最佳")
    print("   - Graphlet信息对结构化网络（如飞行网络）更有效")
    print("   - 计算开销随网络规模和密度显著增加")
    print("   - 增强方法显著改善了嵌入质量和可分离性")
    
    print("=" * 80)

def main():
    """主函数"""
    # 提取数据
    datasets_df = extract_and_create_datasets()
    flight_df = extract_flight_results()
    lastfm_df = extract_social_network_results()
    wiki_df = extract_wiki_results()
    runtime_df = extract_runtime_results()
    pca_df = extract_pca_results()
    
    # 生成可视化
    print("正在生成数据集概览图...")
    plot_dataset_overview(datasets_df)
    
    print("正在生成飞行网络性能图...")
    plot_flight_performance(flight_df)
    
    print("正在生成所有网络性能热图...")
    plot_all_networks_performance(flight_df, lastfm_df, wiki_df)
    
    print("正在生成运行时间分析图...")
    plot_runtime_analysis(runtime_df)
    
    print("正在生成PCA分析图...")
    plot_pca_analysis(pca_df)
    
    # 生成总结报告
    create_summary_report()
    
    print("\n所有图表已保存到当前目录!")
    print("生成的文件:")
    print("- dataset_overview.png: 数据集概览")
    print("- flight_performance_comparison.png: 飞行网络性能对比")
    print("- all_networks_performance_heatmap.png: 所有网络性能热图")
    print("- runtime_analysis.png: 运行时间分析")
    print("- pca_analysis.png: PCA质量分析")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的评估脚本，用于比较不同method的effectiveness
"""
import os
import sys
import numpy as np
import networkx as nx
import pickle
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from GraphEmbedding.ge.classify import read_node_label, Classifier
from algorithms.traditional.struc2vec import Struc2Vec
from algorithms.graphlet_based.compute_edges_improved import (
    generate_improved_structural_distance,
    adaptive_fusion
)


class ImprovedEvaluator:
    """改进的评估器，支持多种分类器和详细指标"""
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        
    def evaluate_with_multiple_classifiers(self, X, Y, tr_frac=0.8, random_state=42):
        """使用多种分类器评估嵌入quality"""
        classifiers = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
            'SVM': SVC(kernel='rbf', probability=True, random_state=random_state),  # 启用概率预测
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state)
        }
        
        results = {}
        
        for clf_name, clf_model in classifiers.items():
            print(f"\n评估 {clf_name}...")
            clf = Classifier(embeddings=self.embeddings, clf=clf_model)
            metrics = clf.split_train_evaluate(X, Y, tr_frac)
            results[clf_name] = metrics
            
        return results
    
    def evaluate_at_different_ratios(self, X, Y, ratios=[0.1, 0.2, 0.4, 0.6, 0.8]):
        """在不同Training比例下评估"""
        results = {}
        
        for ratio in ratios:
            print(f"\nTraining比例: {ratio * 100}%")
            clf = Classifier(embeddings=self.embeddings, clf=LogisticRegression(max_iter=1000))
            metrics = clf.split_train_evaluate(X, Y, ratio)
            results[ratio] = metrics
            
        return results
    
    def visualize_embeddings(self, X, Y, title="Embedding Visualization"):
        """可视化嵌入"""
        emb_list = []
        labels = []
        
        for i, node in enumerate(X):
            if node in self.embeddings:
                emb_list.append(self.embeddings[node])
                labels.append(Y[i][0])
        
        emb_array = np.array(emb_list)
        
        # t-SNE 降维
        tsne = TSNE(n_components=2, random_state=42)
        emb_2d = tsne.fit_transform(emb_array)
        
        # 绘图
        plt.figure(figsize=(10, 8))
        unique_labels = list(set(labels))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = [l == label for l in labels]
            plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], 
                       c=[color], label=label, alpha=0.6)
        
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        return plt


def compare_methods(graph_path, label_path, output_dir="output/"):
    """比较不同method的performance"""
    print("=" * 80)
    print("Starting比较不同的 struc2vec 变体")
    print("=" * 80)
    
    # Load graph - 使用字符串nodes类型以与Label file匹配
    G = nx.read_edgelist(graph_path, nodetype=str, create_using=nx.DiGraph())
        
    print(f"图加载Success: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"nodes类型示例: {type(list(G.nodes())[0])}")
    
    # 准备results存储
    all_results = {}
    
    # 1. 原始 struc2vec（基于度序列）
    print("\n1. Training原始 struc2vec...")
    start_time = time.time()
    model_original = Struc2Vec(G, 10, 80, workers=4, verbose=10, 
                              opt1_reduce_len=True, opt2_reduce_sim_calc=True)
    model_original.train()
    embeddings_original = model_original.get_embeddings()
    time_original = time.time() - start_time
    
    # 评估
    evaluator = ImprovedEvaluator(embeddings_original)
    X, Y = read_node_label(label_path, skip_head=True)
    results_original = evaluator.evaluate_with_multiple_classifiers(X, Y)
    all_results['Original'] = {
        'metrics': results_original,
        'time': time_original
    }
    
    # 2. Graphlet 增强版（简单融合）
    print("\n2. Training Graphlet 增强版（简单融合）...")
    start_time = time.time()
    model_simple = Struc2Vec(G, 10, 80, workers=4, verbose=10,
                            opt1_reduce_len=True, opt2_reduce_sim_calc=True,
                            structural_dist_file=f"{output_dir}structural_dist_brazil-airports.pkl")
    model_simple.train()
    embeddings_simple = model_simple.get_embeddings()
    time_simple = time.time() - start_time
    
    evaluator = ImprovedEvaluator(embeddings_simple)
    results_simple = evaluator.evaluate_with_multiple_classifiers(X, Y)
    all_results['Graphlet_Simple'] = {
        'metrics': results_simple,
        'time': time_simple
    }
    
    # 3. 改进的 Graphlet 增强版
    print("\n3. Generating改进的 Graphlet 距离...")
    improved_dist_path = f"{output_dir}structural_dist_improved.pkl"
    
    # Generating改进的距离
    prefix = graph_path.split('/')[-1].split('.')[0]
    generate_improved_structural_distance(
        graph_path,
        improved_dist_path,
        max_layer=5,
        distance_method='combined',
        use_orbit_selection=True,
        top_k_orbits=40
    )
    
    print("\nTraining改进的 Graphlet 增强版...")
    start_time = time.time()
    model_improved = Struc2Vec(G, 10, 80, workers=4, verbose=10,
                              opt1_reduce_len=True, opt2_reduce_sim_calc=True,
                              structural_dist_file=improved_dist_path)
    model_improved.train()
    embeddings_improved = model_improved.get_embeddings()
    time_improved = time.time() - start_time
    
    evaluator = ImprovedEvaluator(embeddings_improved)
    results_improved = evaluator.evaluate_with_multiple_classifiers(X, Y)
    all_results['Graphlet_Improved'] = {
        'metrics': results_improved,
        'time': time_improved
    }
    
    # 4. 自适应融合版
    print("\n4. Training自适应融合版...")
    # 加载原始距离和 graphlet 距离
    with open(f"{output_dir}structural_dist.pkl", 'rb') as f:
        dist_original = pickle.load(f)
    with open(improved_dist_path, 'rb') as f:
        dist_improved = pickle.load(f)
    
    # 自适应融合
    try:
        dist_adaptive = adaptive_fusion(dist_improved, dist_original, G)
    except Exception as e:
        print(f"自适应融合Failed: {e}")
        print("跳过自适应融合版本...")
        return all_results
    adaptive_path = f"{output_dir}structural_dist_adaptive.pkl"
    with open(adaptive_path, 'wb') as f:
        pickle.dump(dist_adaptive, f)
    
    start_time = time.time()
    model_adaptive = Struc2Vec(G, 10, 80, workers=4, verbose=10,
                              opt1_reduce_len=True, opt2_reduce_sim_calc=True,
                              structural_dist_file=adaptive_path)
    model_adaptive.train()
    embeddings_adaptive = model_adaptive.get_embeddings()
    time_adaptive = time.time() - start_time
    
    evaluator = ImprovedEvaluator(embeddings_adaptive)
    results_adaptive = evaluator.evaluate_with_multiple_classifiers(X, Y)
    all_results['Graphlet_Adaptive'] = {
        'metrics': results_adaptive,
        'time': time_adaptive
    }
    
    # 打印results摘要
    print("\n" + "=" * 80)
    print("performance比较摘要")
    print("=" * 80)
    
    for method, data in all_results.items():
        print(f"\n{method}:")
        print(f"  TrainingTime: {data['time']:.2f} 秒")
        for clf_name, metrics in data['metrics'].items():
            print(f"  {clf_name}:")
            print(f"    - Accuracy: {metrics['acc']:.4f}")
            print(f"    - F1 Micro: {metrics['micro']:.4f}")
            print(f"    - F1 Macro: {metrics['macro']:.4f}")
    
    # 可视化比较
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (method, embeddings) in enumerate([
        ('Original', embeddings_original),
        ('Graphlet Simple', embeddings_simple),
        ('Graphlet Improved', embeddings_improved),
        ('Graphlet Adaptive', embeddings_adaptive)
    ]):
        ax = axes[idx]
        evaluator = ImprovedEvaluator(embeddings)
        
        # 在子图中可视化
        emb_list = []
        labels = []
        for i, node in enumerate(X):
            if node in embeddings:
                emb_list.append(embeddings[node])
                labels.append(Y[i][0])
        
        if emb_list:
            emb_array = np.array(emb_list)
            tsne = TSNE(n_components=2, random_state=42)
            emb_2d = tsne.fit_transform(emb_array)
            
            unique_labels = list(set(labels))
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                mask = [l == label for l in labels]
                ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], 
                          c=[color], label=label, alpha=0.6)
            
            ax.set_title(f'{method} Embeddings')
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}comparison_visualization.png", dpi=300)
    plt.show()
    
    # Training比例实验
    print("\n" + "=" * 80)
    print("不同Training比例下的performance")
    print("=" * 80)
    
    ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    ratio_results = {}
    
    for method, embeddings in [
        ('Original', embeddings_original),
        ('Improved', embeddings_improved),
        ('Adaptive', embeddings_adaptive)
    ]:
        print(f"\n{method}:")
        evaluator = ImprovedEvaluator(embeddings)
        ratio_results[method] = evaluator.evaluate_at_different_ratios(X, Y, ratios)
    
    # 绘制Training比例曲线
    plt.figure(figsize=(10, 6))
    for method, results in ratio_results.items():
        accuracies = [results[r]['acc'] for r in ratios]
        plt.plot(ratios, accuracies, marker='o', label=method)
    
    plt.xlabel('Training Ratio')
    plt.ylabel('Accuracy')
    plt.title('Performance vs Training Ratio')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}training_ratio_comparison.png", dpi=300)
    plt.show()
    
    return all_results


if __name__ == "__main__":
    # 运行完整比较
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results = compare_methods(
        os.path.join(base_dir, "data/flight/brazil-airports.edgelist"),
        os.path.join(base_dir, "data/flight/labels-brazil-airports.txt")
    )
    
    # 也可以测试其他dataset
    # results = compare_methods(
    #     "../../data/wiki/Wiki_edgelist.txt",
    #     "../../data/wiki/wiki_labels.txt"
    # )
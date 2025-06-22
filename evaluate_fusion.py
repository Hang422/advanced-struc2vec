#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估特征融合方法
"""
import os
import sys
import time
import pickle
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from GraphEmbedding.ge.classify import read_node_label, Classifier
from algorithms.traditional.struc2vec import Struc2Vec
from algorithms.graphlet_based.compute_edges_improved import generate_improved_structural_distance

def generate_original_structural_distance(graph_path, output_path):
    """生成原始 struc2vec 的结构距离文件"""
    print("生成原始结构距离...")
    
    # 创建临时的 struc2vec 对象来生成距离
    G = nx.read_edgelist(graph_path, nodetype=str, create_using=nx.DiGraph())
    model = Struc2Vec(G, walk_length=10, num_walks=1, workers=1, verbose=0,
                     opt1_reduce_len=True, opt2_reduce_sim_calc=True)
    
    # 距离文件会在临时目录中生成
    temp_dist_file = os.path.join(model.temp_path, "structural_dist.pkl")
    
    # 复制到输出路径
    import shutil
    shutil.copy(temp_dist_file, output_path)
    
    # 清理临时文件
    shutil.rmtree(model.temp_path)
    
    print(f"✅ 原始距离文件生成: {output_path}")

def fuse_distances(dist1_path, dist2_path, output_path, method='weighted', alpha=0.5):
    """融合两个距离文件"""
    with open(dist1_path, 'rb') as f:
        dist1 = pickle.load(f)
    with open(dist2_path, 'rb') as f:
        dist2 = pickle.load(f)
    
    fused = {}
    
    if method == 'weighted':
        # 加权平均融合
        for pair in set(dist1.keys()).intersection(dist2.keys()):
            fused[pair] = {}
            layers1 = dist1[pair]
            layers2 = dist2[pair]
            for layer in set(layers1.keys()).intersection(layers2.keys()):
                fused[pair][layer] = alpha * layers1[layer] + (1 - alpha) * layers2[layer]
                
    elif method == 'min':
        # 取最小值融合
        for pair in set(dist1.keys()).intersection(dist2.keys()):
            fused[pair] = {}
            layers1 = dist1[pair]
            layers2 = dist2[pair]
            for layer in set(layers1.keys()).intersection(layers2.keys()):
                fused[pair][layer] = min(layers1[layer], layers2[layer])
                
    elif method == 'max':
        # 取最大值融合
        for pair in set(dist1.keys()).intersection(dist2.keys()):
            fused[pair] = {}
            layers1 = dist1[pair]
            layers2 = dist2[pair]
            for layer in set(layers1.keys()).intersection(layers2.keys()):
                fused[pair][layer] = max(layers1[layer], layers2[layer])
                
    elif method == 'adaptive':
        # 自适应融合（根据层次调整权重）
        for pair in set(dist1.keys()).intersection(dist2.keys()):
            fused[pair] = {}
            layers1 = dist1[pair]
            layers2 = dist2[pair]
            for layer in set(layers1.keys()).intersection(layers2.keys()):
                # 低层更信任度序列，高层更信任 graphlet
                layer_alpha = min(0.8, 0.3 + 0.1 * layer)
                fused[pair][layer] = layer_alpha * layers2[layer] + (1 - layer_alpha) * layers1[layer]
    
    with open(output_path, 'wb') as f:
        pickle.dump(fused, f)
    
    return fused

def evaluate_fusion_methods():
    """评估不同的融合方法"""
    print("=" * 80)
    print("特征融合方法评估")
    print("=" * 80)
    
    # 设置路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    graph_path = os.path.join(base_dir, "data/flight/brazil-airports.edgelist")
    label_path = os.path.join(base_dir, "data/flight/labels-brazil-airports.txt")
    output_dir = os.path.join(base_dir, "output")
    
    # 距离文件路径
    original_dist = os.path.join(output_dir, "structural_dist_original.pkl")
    graphlet_dist = os.path.join(output_dir, "structural_dist_brazil-airports.pkl")
    
    # 加载数据
    print("\n加载数据...")
    G = nx.read_edgelist(graph_path, nodetype=str, create_using=nx.DiGraph())
    X, Y = read_node_label(label_path, skip_head=True)
    
    print(f"图信息: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    print(f"标签信息: {len(X)} 个标记节点")
    
    # 生成原始距离文件（如果不存在）
    if not os.path.exists(original_dist):
        generate_original_structural_distance(graph_path, original_dist)
    
    # 确保 graphlet 距离文件存在
    if not os.path.exists(graphlet_dist):
        print("\n生成 Graphlet 距离文件...")
        generate_improved_structural_distance(
            graph_path,
            graphlet_dist,
            max_layer=3,
            distance_method='frobenius',
            use_orbit_selection=False
        )
    
    results = {}
    
    # 1. 基准方法
    print("\n1. 评估基准方法...")
    
    # 原始 struc2vec
    print("   - 原始 Struc2Vec...")
    start = time.time()
    model = Struc2Vec(G, walk_length=40, num_walks=10, workers=2, verbose=0,
                     structural_dist_file=original_dist)
    model.train(embed_size=64, window_size=5, workers=2, iter=3)
    embeddings = model.get_embeddings()
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression(max_iter=1000))
    metrics = clf.split_train_evaluate(X, Y, 0.8)
    results['Original'] = {
        'time': time.time() - start,
        'metrics': metrics
    }
    
    # Graphlet 增强版
    print("   - Graphlet 增强版...")
    start = time.time()
    model = Struc2Vec(G, walk_length=40, num_walks=10, workers=2, verbose=0,
                     structural_dist_file=graphlet_dist)
    model.train(embed_size=64, window_size=5, workers=2, iter=3)
    embeddings = model.get_embeddings()
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression(max_iter=1000))
    metrics = clf.split_train_evaluate(X, Y, 0.8)
    results['Graphlet'] = {
        'time': time.time() - start,
        'metrics': metrics
    }
    
    # 2. 融合方法
    fusion_methods = {
        'Weighted (α=0.5)': {'method': 'weighted', 'alpha': 0.5},
        'Weighted (α=0.3)': {'method': 'weighted', 'alpha': 0.3},
        'Weighted (α=0.7)': {'method': 'weighted', 'alpha': 0.7},
        'Min Fusion': {'method': 'min'},
        'Max Fusion': {'method': 'max'},
        'Adaptive': {'method': 'adaptive'}
    }
    
    print("\n2. 评估融合方法...")
    for name, params in fusion_methods.items():
        print(f"   - {name}...")
        
        # 生成融合距离
        fused_path = os.path.join(output_dir, f"structural_dist_fused_{params['method']}.pkl")
        fuse_distances(
            original_dist, 
            graphlet_dist, 
            fused_path,
            method=params['method'],
            alpha=params.get('alpha', 0.5)
        )
        
        # 训练和评估
        start = time.time()
        model = Struc2Vec(G, walk_length=40, num_walks=10, workers=2, verbose=0,
                         structural_dist_file=fused_path)
        model.train(embed_size=64, window_size=5, workers=2, iter=3)
        embeddings = model.get_embeddings()
        clf = Classifier(embeddings=embeddings, clf=LogisticRegression(max_iter=1000))
        metrics = clf.split_train_evaluate(X, Y, 0.8)
        results[name] = {
            'time': time.time() - start,
            'metrics': metrics
        }
    
    # 3. 打印结果比较
    print("\n" + "=" * 80)
    print("结果汇总")
    print("=" * 80)
    
    print(f"\n{'方法':<20} {'准确率':<10} {'F1 Micro':<10} {'F1 Macro':<10} {'训练时间':<10}")
    print("-" * 70)
    
    for method, data in results.items():
        metrics = data['metrics']
        print(f"{method:<20} {metrics['acc']:<10.4f} {metrics['micro']:<10.4f} "
              f"{metrics['macro']:<10.4f} {data['time']:<10.2f}")
    
    # 4. 找出最佳方法
    best_method = max(results.keys(), key=lambda k: results[k]['metrics']['acc'])
    best_acc = results[best_method]['metrics']['acc']
    
    print(f"\n最佳方法: {best_method} (准确率: {best_acc:.4f})")
    
    # 5. 改进分析
    print("\n改进分析:")
    original_acc = results['Original']['metrics']['acc']
    for method, data in results.items():
        if method != 'Original':
            improvement = (data['metrics']['acc'] - original_acc) / original_acc * 100
            print(f"  {method}: {improvement:+.1f}%")
    
    return results

if __name__ == "__main__":
    try:
        results = evaluate_fusion_methods()
        print("\n✅ 融合评估完成!")
        
        # 给出建议
        print("\n💡 建议:")
        print("  1. 如果原始方法已经很好，可以尝试较小的 α 值 (0.2-0.3)")
        print("  2. 自适应融合可能在大图上效果更好")
        print("  3. 可以尝试其他融合策略，如基于节点特性的动态权重")
        
    except Exception as e:
        print(f"\n❌ 出错: {e}")
        import traceback
        traceback.print_exc()
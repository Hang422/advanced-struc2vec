#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的评估脚本 - 比较原始 struc2vec 和 graphlet 增强版
"""
import os
import sys
import time
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from GraphEmbedding.ge.classify import read_node_label, Classifier
from algorithms.traditional.struc2vec import Struc2Vec

def compare_methods():
    """比较原始方法和 graphlet 增强方法"""
    print("=" * 80)
    print("Struc2Vec 方法比较")
    print("=" * 80)
    
    # 设置路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    graph_path = os.path.join(base_dir, "data/flight/brazil-airports.edgelist")
    label_path = os.path.join(base_dir, "data/flight/labels-brazil-airports.txt")
    dist_file = os.path.join(base_dir, "output/structural_dist_brazil-airports.pkl")
    
    # 加载图和标签
    print("\n加载数据...")
    G = nx.read_edgelist(graph_path, nodetype=str, create_using=nx.DiGraph())
    X, Y = read_node_label(label_path, skip_head=True)
    
    print(f"图信息: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    print(f"标签信息: {len(X)} 个标记节点")
    
    results = {}
    
    # 1. 原始 struc2vec
    print("\n1. 训练原始 struc2vec (基于度序列)...")
    start_time = time.time()
    
    model_original = Struc2Vec(G, walk_length=40, num_walks=10, workers=2, verbose=0,
                              opt1_reduce_len=True, opt2_reduce_sim_calc=True)
    model_original.train(embed_size=64, window_size=5, workers=2, iter=3)
    embeddings_original = model_original.get_embeddings()
    
    time_original = time.time() - start_time
    
    # 评估
    clf = Classifier(embeddings=embeddings_original, clf=LogisticRegression(max_iter=1000))
    metrics_original = clf.split_train_evaluate(X, Y, 0.8)
    
    results['Original'] = {
        'time': time_original,
        'metrics': metrics_original,
        'embeddings': embeddings_original
    }
    
    # 2. Graphlet 增强版
    if os.path.exists(dist_file):
        print("\n2. 训练 Graphlet 增强版...")
        start_time = time.time()
        
        model_graphlet = Struc2Vec(G, walk_length=40, num_walks=10, workers=2, verbose=0,
                                  opt1_reduce_len=True, opt2_reduce_sim_calc=True,
                                  structural_dist_file=dist_file)
        model_graphlet.train(embed_size=64, window_size=5, workers=2, iter=3)
        embeddings_graphlet = model_graphlet.get_embeddings()
        
        time_graphlet = time.time() - start_time
        
        # 评估
        clf = Classifier(embeddings=embeddings_graphlet, clf=LogisticRegression(max_iter=1000))
        metrics_graphlet = clf.split_train_evaluate(X, Y, 0.8)
        
        results['Graphlet'] = {
            'time': time_graphlet,
            'metrics': metrics_graphlet,
            'embeddings': embeddings_graphlet
        }
    else:
        print(f"\n跳过 Graphlet 增强版: 距离文件不存在")
        print(f"请先运行: python generate_simple_distance.py")
    
    # 打印结果
    print("\n" + "=" * 80)
    print("结果比较:")
    print("=" * 80)
    
    print(f"\n{'方法':<15} {'训练时间':<10} {'准确率':<10} {'F1 Micro':<10} {'F1 Macro':<10}")
    print("-" * 60)
    
    for method, data in results.items():
        metrics = data['metrics']
        print(f"{method:<15} {data['time']:<10.2f} {metrics['acc']:<10.4f} "
              f"{metrics['micro']:<10.4f} {metrics['macro']:<10.4f}")
    
    # 计算改进
    if 'Graphlet' in results:
        print("\n性能分析:")
        acc_diff = results['Graphlet']['metrics']['acc'] - results['Original']['metrics']['acc']
        time_ratio = results['Graphlet']['time'] / results['Original']['time']
        
        print(f"  准确率变化: {acc_diff:+.4f} ({acc_diff/results['Original']['metrics']['acc']*100:+.1f}%)")
        print(f"  时间比率: {time_ratio:.2f}x")
        
        if acc_diff > 0:
            print(f"  结论: Graphlet 增强版提升了准确率")
        else:
            print(f"  结论: 原始版本表现更好")
    
    return results

if __name__ == "__main__":
    try:
        results = compare_methods()
        print("\n✅ 评估完成!")
    except Exception as e:
        print(f"\n❌ 出错: {e}")
        import traceback
        traceback.print_exc()
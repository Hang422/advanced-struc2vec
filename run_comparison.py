#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的比较脚本 - 只运行可以直接执行的方法
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

def simple_comparison():
    """运行简化的比较"""
    print("=" * 80)
    print("Struc2Vec 简化比较")
    print("=" * 80)
    
    # 设置路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    graph_path = os.path.join(base_dir, "data/flight/brazil-airports.edgelist")
    label_path = os.path.join(base_dir, "data/flight/labels-brazil-airports.txt")
    
    # 加载图和标签
    print("\n加载数据...")
    G = nx.read_edgelist(graph_path, nodetype=str, create_using=nx.DiGraph())
    X, Y = read_node_label(label_path, skip_head=True)
    
    print(f"图信息: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    print(f"标签信息: {len(X)} 个标记节点")
    
    # 训练原始 struc2vec
    print("\n训练原始 struc2vec...")
    start_time = time.time()
    
    # 使用较小的参数以加快训练
    model = Struc2Vec(G, walk_length=40, num_walks=10, workers=2, verbose=0,
                     opt1_reduce_len=True, opt2_reduce_sim_calc=True)
    model.train(embed_size=64, window_size=5, workers=2, iter=3)
    embeddings = model.get_embeddings()
    
    training_time = time.time() - start_time
    
    # 评估
    print("\n评估模型性能...")
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression(max_iter=1000))
    metrics = clf.split_train_evaluate(X, Y, 0.8)
    
    print(f"\n结果:")
    print(f"  训练时间: {training_time:.2f} 秒")
    print(f"  准确率: {metrics['acc']:.4f}")
    print(f"  F1 Micro: {metrics['micro']:.4f}")
    print(f"  F1 Macro: {metrics['macro']:.4f}")
    
    # 展示一些嵌入示例
    print(f"\n嵌入示例 (前5个节点):")
    for i, (node, vec) in enumerate(list(embeddings.items())[:5]):
        print(f"  节点 {node}: 维度={len(vec)}, 前5维={vec[:5]}")
    
    return embeddings, metrics

if __name__ == "__main__":
    try:
        embeddings, metrics = simple_comparison()
        print("\n✅ 比较完成!")
    except Exception as e:
        print(f"\n❌ 出错: {e}")
        import traceback
        traceback.print_exc()
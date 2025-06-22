#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的比较脚本 - 只使用逻辑回归进行评估
"""
import os
import sys
import numpy as np
import networkx as nx
import pickle
import time

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from GraphEmbedding.ge.classify import read_node_label, Classifier
from algorithms.traditional.struc2vec import Struc2Vec
from sklearn.linear_model import LogisticRegression


def simple_compare_methods(dataset_name="brazil-airports"):
    """简单比较不同方法的性能 - 只使用逻辑回归"""
    
    # 设置路径
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if dataset_name == "brazil-airports":
        graph_path = os.path.join(project_root, "data/flight/brazil-airports.edgelist")
        label_path = os.path.join(project_root, "data/flight/labels-brazil-airports.txt")
    else:
        raise ValueError(f"未支持的数据集: {dataset_name}")
    
    output_dir = os.path.join(project_root, "output/")
    
    print(f"{'='*80}")
    print(f"简化的 Struc2Vec 方法比较 - 数据集: {dataset_name}")
    print(f"{'='*80}")
    
    # 加载图和标签
    G = nx.read_edgelist(graph_path, create_using=nx.DiGraph(), 
                        nodetype=None, data=[('weight', int)])
    X, Y = read_node_label(label_path, skip_head=True)
    
    print(f"图信息: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    print(f"标签信息: {len(X)} 个标记节点")
    
    results = {}
    
    # 1. 原始 struc2vec
    print(f"\n1. 训练原始 struc2vec...")
    start_time = time.time()
    
    model = Struc2Vec(G, num_walks=5, walk_length=40, workers=2, verbose=0,
                     opt1_reduce_len=True, opt2_reduce_sim_calc=True)
    model.train(embed_size=64, iter=3)
    embeddings = model.get_embeddings()
    
    training_time = time.time() - start_time
    
    # 评估
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression(max_iter=1000))
    metrics = clf.split_train_evaluate(X, Y, 0.8)
    
    results['Original'] = {
        'time': training_time,
        'accuracy': metrics['acc'],
        'f1_micro': metrics['micro'],
        'f1_macro': metrics['macro']
    }
    
    print(f"   完成: {training_time:.2f}秒, 准确率: {metrics['acc']:.4f}")
    
    # 2. 测试可用的改进版本
    improved_files = {
        'Improved_Basic': os.path.join(output_dir, f"structural_dist_improved_basic_{dataset_name}.pkl"),
        'Improved_Compact': os.path.join(output_dir, f"structural_dist_improved_compact_{dataset_name}.pkl"),
        'Improved_Frobenius': os.path.join(output_dir, f"structural_dist_improved_frobenius_{dataset_name}.pkl")
    }
    
    for method_name, dist_file in improved_files.items():
        if not os.path.exists(dist_file):
            print(f"   跳过 {method_name}: 距离文件不存在")
            continue
            
        print(f"\n2. 训练 {method_name}...")
        start_time = time.time()
        
        model = Struc2Vec(G, num_walks=5, walk_length=40, workers=2, verbose=0,
                         opt1_reduce_len=True, opt2_reduce_sim_calc=True,
                         structural_dist_file=dist_file)
        model.train(embed_size=64, iter=3)
        embeddings = model.get_embeddings()
        
        training_time = time.time() - start_time
        
        # 评估
        clf = Classifier(embeddings=embeddings, clf=LogisticRegression(max_iter=1000))
        metrics = clf.split_train_evaluate(X, Y, 0.8)
        
        results[method_name] = {
            'time': training_time,
            'accuracy': metrics['acc'],
            'f1_micro': metrics['micro'],
            'f1_macro': metrics['macro']
        }
        
        print(f"   完成: {training_time:.2f}秒, 准确率: {metrics['acc']:.4f}")
    
    # 3. 结果比较
    print(f"\n{'='*80}")
    print("结果总结:")
    print(f"{'='*80}")
    
    print(f"{'方法':<20} {'训练时间':<10} {'准确率':<10} {'F1-Micro':<10} {'F1-Macro':<10}")
    print("-" * 70)
    
    for method, result in results.items():
        print(f"{method:<20} {result['time']:<10.2f} {result['accuracy']:<10.4f} "
              f"{result['f1_micro']:<10.4f} {result['f1_macro']:<10.4f}")
    
    # 4. 性能提升分析
    if len(results) > 1:
        print(f"\n性能提升分析 (相比原始方法):")
        original_acc = results['Original']['accuracy']
        
        for method, result in results.items():
            if method != 'Original':
                acc_improvement = (result['accuracy'] - original_acc) / original_acc * 100
                time_ratio = result['time'] / results['Original']['time']
                print(f"  {method}: 准确率提升 {acc_improvement:+.2f}%, 时间比例 {time_ratio:.2f}x")
    
    # 5. 推荐
    print(f"\n推荐:")
    if len(results) > 1:
        best_acc_method = max(results.keys(), key=lambda k: results[k]['accuracy'])
        fastest_method = min(results.keys(), key=lambda k: results[k]['time'])
        
        print(f"  最佳准确率: {best_acc_method} ({results[best_acc_method]['accuracy']:.4f})")
        print(f"  最快训练: {fastest_method} ({results[fastest_method]['time']:.2f}秒)")
        
        # 综合推荐
        scores = {}
        for method, result in results.items():
            # 综合分数: 准确率权重0.7 + 速度权重0.3 (速度取倒数并归一化)
            max_time = max(r['time'] for r in results.values())
            speed_score = (max_time - result['time']) / max_time
            scores[method] = 0.7 * result['accuracy'] + 0.3 * speed_score
        
        best_overall = max(scores.keys(), key=lambda k: scores[k])
        print(f"  综合推荐: {best_overall} (综合分数: {scores[best_overall]:.4f})")
    
    return results


def quick_test_single_method(method_file, dataset_name="brazil-airports"):
    """快速测试单个方法"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    graph_path = os.path.join(project_root, "data/flight/brazil-airports.edgelist")
    label_path = os.path.join(project_root, "data/flight/labels-brazil-airports.txt")
    
    if not os.path.exists(method_file):
        print(f"❌ 方法文件不存在: {method_file}")
        return None
    
    print(f"🚀 快速测试: {os.path.basename(method_file)}")
    
    # 加载数据
    G = nx.read_edgelist(graph_path, create_using=nx.DiGraph(), 
                        nodetype=None, data=[('weight', int)])
    X, Y = read_node_label(label_path, skip_head=True)
    
    # 训练
    start_time = time.time()
    model = Struc2Vec(G, num_walks=3, walk_length=20, workers=1, verbose=0,
                     structural_dist_file=method_file)
    model.train(embed_size=32, iter=1)
    embeddings = model.get_embeddings()
    training_time = time.time() - start_time
    
    # 评估
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression(max_iter=500))
    metrics = clf.split_train_evaluate(X, Y, 0.8)
    
    print(f"   ✅ 完成: {training_time:.2f}秒, 准确率: {metrics['acc']:.4f}")
    
    return {
        'time': training_time,
        'accuracy': metrics['acc'],
        'f1_micro': metrics['micro'],
        'f1_macro': metrics['macro']
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "compare":
            # 完整比较
            dataset = sys.argv[2] if len(sys.argv) > 2 else "brazil-airports"
            results = simple_compare_methods(dataset)
            
        elif command == "test" and len(sys.argv) > 2:
            # 测试单个方法文件
            method_file = sys.argv[2]
            result = quick_test_single_method(method_file)
            
        else:
            print("用法:")
            print("  python simple_compare.py compare [dataset_name]")
            print("  python simple_compare.py test [method_file_path]")
    else:
        # 默认运行完整比较
        print("运行默认比较...")
        results = simple_compare_methods("brazil-airports")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化融合权重
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

def fuse_with_alpha(dist1_path, dist2_path, alpha):
    """使用指定权重融合距离"""
    with open(dist1_path, 'rb') as f:
        dist1 = pickle.load(f)
    with open(dist2_path, 'rb') as f:
        dist2 = pickle.load(f)
    
    fused = {}
    for pair in set(dist1.keys()).intersection(dist2.keys()):
        fused[pair] = {}
        layers1 = dist1[pair]
        layers2 = dist2[pair]
        for layer in set(layers1.keys()).intersection(layers2.keys()):
            fused[pair][layer] = alpha * layers1[layer] + (1 - alpha) * layers2[layer]
    
    return fused

def evaluate_fusion_alpha(G, X, Y, dist1_path, dist2_path, alpha):
    """评估特定权重的融合效果"""
    try:
        # 生成融合距离
        fused_dist = fuse_with_alpha(dist1_path, dist2_path, alpha)
        
        # 保存临时文件
        temp_path = f"temp_fused_{alpha:.3f}.pkl"
        with open(temp_path, 'wb') as f:
            pickle.dump(fused_dist, f)
        
        # 训练模型
        model = Struc2Vec(G, walk_length=40, num_walks=6, workers=1, verbose=0,
                         opt1_reduce_len=True, opt2_reduce_sim_calc=True,
                         structural_dist_file=temp_path)
        model.train(embed_size=64, window_size=5, workers=1, iter=2)
        embeddings = model.get_embeddings()
        
        # 评估
        clf = Classifier(embeddings=embeddings, clf=LogisticRegression(max_iter=1000))
        metrics = clf.split_train_evaluate(X, Y, 0.8)
        
        # 清理临时文件
        os.remove(temp_path)
        
        return metrics['acc']
        
    except Exception as e:
        print(f"   ❌ α={alpha:.3f} 失败: {e}")
        return 0.0

def grid_search_alpha():
    """网格搜索最佳权重"""
    print("=" * 80)
    print("融合权重优化")
    print("=" * 80)
    
    # 设置路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    graph_path = os.path.join(base_dir, "data/flight/brazil-airports.edgelist")
    label_path = os.path.join(base_dir, "data/flight/labels-brazil-airports.txt")
    output_dir = os.path.join(base_dir, "output")
    
    original_dist = os.path.join(output_dir, "structural_dist_original.pkl")
    graphlet_dist = os.path.join(output_dir, "structural_dist_brazil-airports.pkl")
    
    # 加载数据
    print("\n加载数据...")
    G = nx.read_edgelist(graph_path, nodetype=str, create_using=nx.DiGraph())
    X, Y = read_node_label(label_path, skip_head=True)
    
    print(f"图信息: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    print(f"标签信息: {len(X)} 个标记节点")
    
    # 权重候选值
    alphas = np.arange(0.0, 1.1, 0.1)  # 0.0 到 1.0，步长 0.1
    
    print(f"\n开始权重搜索...")
    print(f"搜索范围: α ∈ [0.0, 1.0]，步长 0.1")
    print(f"α=0.0: 纯 Graphlet，α=1.0: 纯度序列")
    
    results = {}
    best_alpha = 0.0
    best_acc = 0.0
    
    for alpha in alphas:
        print(f"\n测试 α={alpha:.1f}...")
        start = time.time()
        
        acc = evaluate_fusion_alpha(G, X, Y, original_dist, graphlet_dist, alpha)
        elapsed = time.time() - start
        
        results[alpha] = {
            'accuracy': acc,
            'time': elapsed
        }
        
        print(f"   准确率: {acc:.4f}, 耗时: {elapsed:.2f}s")
        
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha
            print(f"   🎯 新的最佳权重!")
    
    # 打印结果
    print("\n" + "=" * 80)
    print("权重优化结果")
    print("=" * 80)
    
    print(f"\n{'权重 α':<10} {'准确率':<10} {'时间(s)':<10} {'说明':<20}")
    print("-" * 55)
    
    for alpha in sorted(results.keys()):
        data = results[alpha]
        marker = "🏆" if alpha == best_alpha else "  "
        desc = get_alpha_description(alpha)
        print(f"{marker} {alpha:<8.1f} {data['accuracy']:<10.4f} {data['time']:<10.2f} {desc}")
    
    print(f"\n🎯 最佳权重: α = {best_alpha:.1f}")
    print(f"🎯 最佳准确率: {best_acc:.4f}")
    
    # 保存最佳融合文件
    print(f"\n保存最佳融合文件...")
    best_fused = fuse_with_alpha(original_dist, graphlet_dist, best_alpha)
    best_path = os.path.join(output_dir, f"fused_optimal_alpha_{best_alpha:.1f}.pkl")
    with open(best_path, 'wb') as f:
        pickle.dump(best_fused, f)
    print(f"✅ 最佳融合文件: {best_path}")
    
    # 分析趋势
    print(f"\n📈 趋势分析:")
    accuracies = [results[alpha]['accuracy'] for alpha in sorted(results.keys())]
    
    print(f"  纯 Graphlet (α=0.0): {results[0.0]['accuracy']:.4f}")
    print(f"  均衡融合 (α=0.5): {results[0.5]['accuracy']:.4f}")
    print(f"  纯度序列 (α=1.0): {results[1.0]['accuracy']:.4f}")
    
    # 寻找趋势
    max_idx = accuracies.index(max(accuracies))
    optimal_range = f"α ∈ [{max(0, (max_idx-1)*0.1):.1f}, {min(1.0, (max_idx+1)*0.1):.1f}]"
    print(f"  最优区间: {optimal_range}")
    
    return results, best_alpha, best_acc

def get_alpha_description(alpha):
    """获取权重的描述"""
    if alpha == 0.0:
        return "纯 Graphlet"
    elif alpha == 1.0:
        return "纯度序列"
    elif alpha == 0.5:
        return "均衡融合"
    elif alpha < 0.5:
        return "Graphlet 为主"
    else:
        return "度序列为主"

def fine_search(best_alpha, results):
    """在最佳权重附近进行精细搜索"""
    print(f"\n" + "=" * 60)
    print(f"精细搜索 (α = {best_alpha:.1f} ± 0.1)")
    print("=" * 60)
    
    # 在最佳权重附近搜索
    fine_alphas = np.arange(max(0, best_alpha-0.1), min(1.1, best_alpha+0.15), 0.02)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    graph_path = os.path.join(base_dir, "data/flight/brazil-airports.edgelist")
    label_path = os.path.join(base_dir, "data/flight/labels-brazil-airports.txt")
    output_dir = os.path.join(base_dir, "output")
    
    original_dist = os.path.join(output_dir, "structural_dist_original.pkl")
    graphlet_dist = os.path.join(output_dir, "structural_dist_brazil-airports.pkl")
    
    G = nx.read_edgelist(graph_path, nodetype=str, create_using=nx.DiGraph())
    X, Y = read_node_label(label_path, skip_head=True)
    
    fine_best_alpha = best_alpha
    fine_best_acc = results[best_alpha]['accuracy']
    
    print(f"搜索范围: {fine_alphas[0]:.2f} 到 {fine_alphas[-1]:.2f}，步长 0.02")
    
    for alpha in fine_alphas:
        if alpha in results:  # 跳过已经测试过的
            continue
            
        print(f"精细测试 α={alpha:.2f}...")
        acc = evaluate_fusion_alpha(G, X, Y, original_dist, graphlet_dist, alpha)
        
        if acc > fine_best_acc:
            fine_best_acc = acc
            fine_best_alpha = alpha
            print(f"   🎯 精细搜索新最佳: α={alpha:.2f}, 准确率={acc:.4f}")
    
    return fine_best_alpha, fine_best_acc

if __name__ == "__main__":
    try:
        # 粗搜索
        results, best_alpha, best_acc = grid_search_alpha()
        
        # 精细搜索
        fine_alpha, fine_acc = fine_search(best_alpha, results)
        
        print(f"\n" + "=" * 80)
        print("最终优化结果")
        print("=" * 80)
        print(f"🏆 最终最佳权重: α = {fine_alpha:.2f}")
        print(f"🏆 最终最佳准确率: {fine_acc:.4f}")
        
        # 与基准比较
        baseline_acc = 0.6429  # 原始方法的大致准确率
        improvement = (fine_acc - baseline_acc) / baseline_acc * 100
        print(f"📈 相对原始方法改进: {improvement:+.1f}%")
        
        print(f"\n💡 建议:")
        print(f"  1. 使用权重 α = {fine_alpha:.2f} 进行融合")
        print(f"  2. 在其他数据集上验证这个权重")
        print(f"  3. 考虑自适应权重策略")
        
    except Exception as e:
        print(f"\n❌ 出错: {e}")
        import traceback
        traceback.print_exc()
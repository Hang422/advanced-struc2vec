#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的特征融合评估
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

def fuse_distances_simple(dist1_path, dist2_path, output_path, alpha=0.5):
    """简单的加权融合"""
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
    
    with open(output_path, 'wb') as f:
        pickle.dump(fused, f)
    
    print(f"✅ 融合距离文件生成: {output_path}")

def evaluate_method(G, X, Y, dist_file, method_name):
    """评估单个方法"""
    try:
        print(f"   评估 {method_name}...")
        start = time.time()
        
        if dist_file is None:
            # 原始方法，不使用距离文件
            model = Struc2Vec(G, walk_length=40, num_walks=8, workers=1, verbose=0,
                             opt1_reduce_len=True, opt2_reduce_sim_calc=True)
        else:
            # 使用距离文件
            model = Struc2Vec(G, walk_length=40, num_walks=8, workers=1, verbose=0,
                             opt1_reduce_len=True, opt2_reduce_sim_calc=True,
                             structural_dist_file=dist_file)
        
        model.train(embed_size=64, window_size=5, workers=1, iter=3)
        embeddings = model.get_embeddings()
        
        clf = Classifier(embeddings=embeddings, clf=LogisticRegression(max_iter=1000))
        metrics = clf.split_train_evaluate(X, Y, 0.8)
        
        elapsed = time.time() - start
        
        return {
            'time': elapsed,
            'metrics': metrics,
            'success': True
        }
        
    except Exception as e:
        print(f"     ❌ 评估失败: {e}")
        return {
            'time': 0,
            'metrics': {'acc': 0, 'micro': 0, 'macro': 0},
            'success': False
        }

def main():
    """主评估函数"""
    print("=" * 80)
    print("简化特征融合评估")
    print("=" * 80)
    
    # 设置路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    graph_path = os.path.join(base_dir, "data/flight/brazil-airports.edgelist")
    label_path = os.path.join(base_dir, "data/flight/labels-brazil-airports.txt")
    output_dir = os.path.join(base_dir, "output")
    
    graphlet_dist = os.path.join(output_dir, "structural_dist_brazil-airports.pkl")
    
    # 加载数据
    print("\n加载数据...")
    G = nx.read_edgelist(graph_path, nodetype=str, create_using=nx.DiGraph())
    X, Y = read_node_label(label_path, skip_head=True)
    
    print(f"图信息: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    print(f"标签信息: {len(X)} 个标记节点")
    
    # 确保 graphlet 距离文件存在
    if not os.path.exists(graphlet_dist):
        print(f"\n❌ Graphlet 距离文件不存在: {graphlet_dist}")
        print("请先运行: python generate_simple_distance.py")
        return
    
    results = {}
    
    print("\n开始评估...")
    
    # 1. 原始方法
    results['Original'] = evaluate_method(G, X, Y, None, "原始 Struc2Vec")
    
    # 2. Graphlet 方法
    results['Graphlet'] = evaluate_method(G, X, Y, graphlet_dist, "Graphlet 增强版")
    
    # 3. 融合方法
    fusion_configs = [
        (0.8, "度序列为主 (α=0.8)"),
        (0.5, "均衡融合 (α=0.5)"),
        (0.2, "Graphlet为主 (α=0.2)")
    ]
    
    for alpha, name in fusion_configs:
        fused_path = os.path.join(output_dir, f"fused_alpha_{alpha:.1f}.pkl")
        
        # 生成原始距离用于融合
        print(f"\n生成融合距离文件 (α={alpha})...")
        
        try:
            # 创建临时原始距离
            temp_model = Struc2Vec(G, walk_length=10, num_walks=1, workers=1, verbose=0)
            temp_dist_file = os.path.join(temp_model.temp_path, "structural_dist.pkl")
            
            # 等待文件生成
            import time
            time.sleep(1)
            
            if os.path.exists(temp_dist_file):
                fuse_distances_simple(temp_dist_file, graphlet_dist, fused_path, alpha)
                results[name] = evaluate_method(G, X, Y, fused_path, name)
                
                # 清理临时文件
                import shutil
                shutil.rmtree(temp_model.temp_path)
            else:
                print(f"     ❌ 无法生成临时距离文件")
                results[name] = {'success': False, 'metrics': {'acc': 0, 'micro': 0, 'macro': 0}, 'time': 0}
                
        except Exception as e:
            print(f"     ❌ 融合失败: {e}")
            results[name] = {'success': False, 'metrics': {'acc': 0, 'micro': 0, 'macro': 0}, 'time': 0}
    
    # 打印结果
    print("\n" + "=" * 80)
    print("评估结果汇总")
    print("=" * 80)
    
    print(f"\n{'方法':<25} {'状态':<8} {'准确率':<10} {'F1 Micro':<10} {'F1 Macro':<10} {'时间(s)':<10}")
    print("-" * 85)
    
    successful_results = {}
    
    for method, data in results.items():
        status = "✅" if data['success'] else "❌"
        metrics = data['metrics']
        print(f"{method:<25} {status:<8} {metrics['acc']:<10.4f} {metrics['micro']:<10.4f} "
              f"{metrics['macro']:<10.4f} {data['time']:<10.2f}")
        
        if data['success']:
            successful_results[method] = data
    
    # 分析最佳方法
    if successful_results:
        print(f"\n分析:")
        best_method = max(successful_results.keys(), key=lambda k: successful_results[k]['metrics']['acc'])
        best_acc = successful_results[best_method]['metrics']['acc']
        print(f"  最佳方法: {best_method} (准确率: {best_acc:.4f})")
        
        if 'Original' in successful_results:
            baseline = successful_results['Original']['metrics']['acc']
            print(f"  相对于原始方法的改进:")
            for method, data in successful_results.items():
                if method != 'Original':
                    improvement = (data['metrics']['acc'] - baseline) / baseline * 100
                    print(f"    {method}: {improvement:+.1f}%")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
        print("\n✅ 评估完成!")
        
        print("\n💡 下一步建议:")
        print("  1. 尝试调整融合权重 α")
        print("  2. 在更大的数据集上测试")
        print("  3. 尝试其他融合策略（如自适应权重）")
        
    except Exception as e:
        print(f"\n❌ 出错: {e}")
        import traceback
        traceback.print_exc()
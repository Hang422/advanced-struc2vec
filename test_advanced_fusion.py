#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试高级融合方法
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

def test_method(G, X, Y, dist_file, method_name):
    """测试单个方法"""
    try:
        print(f"   测试 {method_name}...")
        start = time.time()
        
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
        print(f"     ❌ 测试失败: {e}")
        return {
            'time': 0,
            'metrics': {'acc': 0, 'micro': 0, 'macro': 0},
            'success': False
        }

def main():
    """主测试函数"""
    print("=" * 80)
    print("高级融合方法评估")
    print("=" * 80)
    
    # 设置路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    graph_path = os.path.join(base_dir, "data/flight/brazil-airports.edgelist")
    label_path = os.path.join(base_dir, "data/flight/labels-brazil-airports.txt")
    output_dir = os.path.join(base_dir, "output")
    
    # 加载数据
    print("\n加载数据...")
    G = nx.read_edgelist(graph_path, nodetype=str, create_using=nx.DiGraph())
    X, Y = read_node_label(label_path, skip_head=True)
    
    print(f"图信息: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    print(f"标签信息: {len(X)} 个标记节点")
    
    # 测试方法列表
    methods = {
        "原始 Struc2Vec": None,
        "Graphlet 增强版": os.path.join(output_dir, "structural_dist_brazil-airports.pkl"),
        "基础融合 (α=0.5)": os.path.join(output_dir, "fused_alpha_0.5.pkl"),
        "自适应融合": os.path.join(output_dir, "fused_adaptive.pkl"),
        "集成融合": os.path.join(output_dir, "fused_ensemble.pkl"),
        "置信度融合": os.path.join(output_dir, "fused_confidence.pkl")
    }
    
    print("\n开始测试...")
    results = {}
    
    for method_name, dist_file in methods.items():
        if dist_file and not os.path.exists(dist_file):
            print(f"   跳过 {method_name}: 距离文件不存在")
            continue
            
        if method_name == "原始 Struc2Vec":
            # 原始方法
            try:
                print(f"   测试 {method_name}...")
                start = time.time()
                
                model = Struc2Vec(G, walk_length=40, num_walks=8, workers=1, verbose=0,
                                 opt1_reduce_len=True, opt2_reduce_sim_calc=True)
                model.train(embed_size=64, window_size=5, workers=1, iter=3)
                embeddings = model.get_embeddings()
                
                clf = Classifier(embeddings=embeddings, clf=LogisticRegression(max_iter=1000))
                metrics = clf.split_train_evaluate(X, Y, 0.8)
                
                elapsed = time.time() - start
                
                results[method_name] = {
                    'time': elapsed,
                    'metrics': metrics,
                    'success': True
                }
                
            except Exception as e:
                print(f"     ❌ 测试失败: {e}")
                results[method_name] = {
                    'time': 0,
                    'metrics': {'acc': 0, 'micro': 0, 'macro': 0},
                    'success': False
                }
        else:
            results[method_name] = test_method(G, X, Y, dist_file, method_name)
    
    # 打印结果
    print("\n" + "=" * 80)
    print("高级融合方法评估结果")
    print("=" * 80)
    
    print(f"\n{'方法':<20} {'状态':<8} {'准确率':<10} {'F1 Micro':<10} {'F1 Macro':<10} {'时间(s)':<10}")
    print("-" * 75)
    
    successful_results = {}
    
    for method, data in results.items():
        status = "✅" if data['success'] else "❌"
        metrics = data['metrics']
        print(f"{method:<20} {status:<8} {metrics['acc']:<10.4f} {metrics['micro']:<10.4f} "
              f"{metrics['macro']:<10.4f} {data['time']:<10.2f}")
        
        if data['success']:
            successful_results[method] = data
    
    # 详细分析
    if successful_results:
        print(f"\n" + "=" * 60)
        print("详细分析")
        print("=" * 60)
        
        # 找出最佳方法
        best_method = max(successful_results.keys(), 
                         key=lambda k: successful_results[k]['metrics']['acc'])
        best_acc = successful_results[best_method]['metrics']['acc']
        print(f"\n🎯 最佳方法: {best_method}")
        print(f"   准确率: {best_acc:.4f}")
        
        # 与原始方法比较
        if '原始 Struc2Vec' in successful_results:
            baseline = successful_results['原始 Struc2Vec']['metrics']['acc']
            print(f"\n📊 相对原始方法的改进:")
            
            improvements = []
            for method, data in successful_results.items():
                if method != '原始 Struc2Vec':
                    improvement = (data['metrics']['acc'] - baseline) / baseline * 100
                    improvements.append((method, improvement))
                    status = "📈" if improvement > 0 else "📉" if improvement < -1 else "➡️"
                    print(f"   {status} {method}: {improvement:+.1f}%")
            
            # 排序显示
            improvements.sort(key=lambda x: x[1], reverse=True)
            if improvements:
                print(f"\n🏆 改进排行:")
                for i, (method, imp) in enumerate(improvements[:3], 1):
                    print(f"   {i}. {method}: {imp:+.1f}%")
        
        # 效率分析
        print(f"\n⚡ 效率分析:")
        for method, data in successful_results.items():
            speed_factor = successful_results['原始 Struc2Vec']['time'] / data['time'] if data['time'] > 0 else 0
            print(f"   {method}: {data['time']:.2f}s ({speed_factor:.1f}x)")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
        print(f"\n✅ 高级融合评估完成!")
        
        print(f"\n💡 总结与建议:")
        print(f"  1. 自适应融合考虑了节点的聚类和度特性")
        print(f"  2. 集成融合组合了多种距离度量策略")
        print(f"  3. 置信度融合基于距离分布的可靠性")
        print(f"  4. 可以进一步调优参数或尝试其他图数据集")
        
    except Exception as e:
        print(f"\n❌ 出错: {e}")
        import traceback
        traceback.print_exc()
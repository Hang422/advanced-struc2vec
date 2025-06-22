#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试现有的融合文件
"""
import os
import sys
import time
import networkx as nx
from sklearn.linear_model import LogisticRegression

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from GraphEmbedding.ge.classify import read_node_label, Classifier
from algorithms.traditional.struc2vec import Struc2Vec

def test_single_method(G, X, Y, dist_file, method_name):
    """测试单个方法"""
    try:
        print(f"测试 {method_name}...")
        start = time.time()
        
        if dist_file is None:
            model = Struc2Vec(G, walk_length=40, num_walks=8, workers=1, verbose=0,
                             opt1_reduce_len=True, opt2_reduce_sim_calc=True)
        else:
            model = Struc2Vec(G, walk_length=40, num_walks=8, workers=1, verbose=0,
                             opt1_reduce_len=True, opt2_reduce_sim_calc=True,
                             structural_dist_file=dist_file)
        
        model.train(embed_size=64, window_size=5, workers=1, iter=3)
        embeddings = model.get_embeddings()
        
        clf = Classifier(embeddings=embeddings, clf=LogisticRegression(max_iter=1000))
        metrics = clf.split_train_evaluate(X, Y, 0.8)
        
        elapsed = time.time() - start
        print(f"✅ {method_name}: 准确率={metrics['acc']:.4f}, 时间={elapsed:.2f}s")
        
        return {
            'accuracy': metrics['acc'],
            'f1_micro': metrics['micro'], 
            'f1_macro': metrics['macro'],
            'time': elapsed,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ {method_name} 失败: {e}")
        return {
            'accuracy': 0.0,
            'f1_micro': 0.0,
            'f1_macro': 0.0, 
            'time': 0.0,
            'success': False
        }

def main():
    """主测试函数"""
    print("=" * 80)
    print("现有融合文件测试")
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
    
    # 测试方法
    methods = [
        ("原始 Struc2Vec", None),
        ("Graphlet 增强版", os.path.join(output_dir, "structural_dist_brazil-airports.pkl")),
        ("融合 α=0.8 (度序列为主)", os.path.join(output_dir, "fused_alpha_0.8.pkl")),
        ("融合 α=0.5 (均衡)", os.path.join(output_dir, "fused_alpha_0.5.pkl")),
        ("融合 α=0.2 (Graphlet为主)", os.path.join(output_dir, "fused_alpha_0.2.pkl"))
    ]
    
    results = {}
    
    print(f"\n开始测试...")
    print("-" * 80)
    
    for method_name, dist_file in methods:
        if dist_file and not os.path.exists(dist_file):
            print(f"⏭️  跳过 {method_name}: 文件不存在")
            continue
            
        result = test_single_method(G, X, Y, dist_file, method_name)
        results[method_name] = result
    
    # 结果汇总
    print(f"\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    
    print(f"\n{'方法':<25} {'准确率':<10} {'F1 Micro':<10} {'F1 Macro':<10} {'时间(s)':<8}")
    print("-" * 75)
    
    successful_results = {}
    for method, data in results.items():
        if data['success']:
            successful_results[method] = data
            print(f"{method:<25} {data['accuracy']:<10.4f} {data['f1_micro']:<10.4f} "
                  f"{data['f1_macro']:<10.4f} {data['time']:<8.2f}")
    
    if successful_results:
        # 找出最佳方法
        best_method = max(successful_results.keys(), 
                         key=lambda k: successful_results[k]['accuracy'])
        best_acc = successful_results[best_method]['accuracy']
        
        print(f"\n🎯 最佳方法: {best_method}")
        print(f"🎯 最佳准确率: {best_acc:.4f}")
        
        # 与原始方法比较
        if '原始 Struc2Vec' in successful_results:
            baseline = successful_results['原始 Struc2Vec']['accuracy']
            print(f"\n📊 相对原始方法的改进:")
            
            improvements = []
            for method, data in successful_results.items():
                if method != '原始 Struc2Vec':
                    improvement = (data['accuracy'] - baseline) / baseline * 100
                    improvements.append((method, improvement))
                    emoji = "📈" if improvement > 0 else "📉"
                    print(f"  {emoji} {method}: {improvement:+.1f}%")
            
            # 找出最佳改进
            if improvements:
                improvements.sort(key=lambda x: x[1], reverse=True)
                print(f"\n🏆 改进排行:")
                for i, (method, imp) in enumerate(improvements, 1):
                    print(f"  {i}. {method}: {imp:+.1f}%")
        
        # 效率分析
        print(f"\n⚡ 效率分析:")
        baseline_time = successful_results.get('原始 Struc2Vec', {}).get('time', 1)
        for method, data in successful_results.items():
            speedup = baseline_time / data['time'] if data['time'] > 0 else 0
            print(f"  {method}: {data['time']:.2f}s ({speedup:.1f}x)")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
        print(f"\n✅ 测试完成!")
        
        # 给出具体建议
        print(f"\n💡 具体建议:")
        print(f"  1. 如果基础融合效果好，可以尝试更多权重值")
        print(f"  2. 在更大的数据集上验证融合效果")
        print(f"  3. 考虑任务特定的融合策略")
        
    except Exception as e:
        print(f"\n❌ 出错: {e}")
        import traceback
        traceback.print_exc()
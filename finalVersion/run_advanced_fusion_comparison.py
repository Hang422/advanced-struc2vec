#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级融合方法比较脚本
测试和比较多种先进的特征融合技术
"""
import sys
import os
import time
import argparse
import numpy as np
import networkx as nx
from pathlib import Path
from sklearn.linear_model import LogisticRegression

# 添加父项目路径
parent_path = Path(__file__).parent.parent
sys.path.append(str(parent_path))

# 应用 Struc2Vec 除零警告修复补丁
try:
    from src.utils import struc2vec_patch
except:
    pass  # 补丁是可选的

from libs.GraphEmbedding.ge.classify import read_node_label, Classifier
from libs.GraphEmbedding.ge.models.struc2vec import Struc2Vec
from src.algorithms.advanced_fusion_methods import AdvancedFusionStruc2Vec

def load_dataset(dataset_name):
    """加载数据集"""
    data_base = parent_path / 'data'
    
    datasets = {
        'usa-airports': {
            'graph': data_base / 'flight/usa-airports.edgelist',
            'labels': data_base / 'flight/labels-usa-airports.txt'
        },
        'wiki': {
            'graph': data_base / 'wiki/Wiki_edgelist.txt',
            'labels': data_base / 'wiki/wiki_labels.txt'
        },
        'lastfm': {
            'graph': data_base / 'lastfm_asia/lastfm_asia.edgelist',
            'labels': data_base / 'lastfm_asia/lastfm_asia_labels.txt'
        }
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"未知数据集: {dataset_name}")
    
    info = datasets[dataset_name]
    
    print(f"📂 加载数据集: {dataset_name}")
    print(f"   图文件: {info['graph']}")
    print(f"   标签文件: {info['labels']}")
    
    # 加载图
    G = nx.read_edgelist(str(info['graph']), nodetype=str, create_using=nx.DiGraph())
    print(f"   图信息: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    
    # 加载标签
    X, Y = read_node_label(str(info['labels']), skip_head=True)
    print(f"   标签信息: {len(X)} 个标记节点")
    
    return G, X, Y

def evaluate_method(embeddings, X, Y, method_name):
    """评估单个方法"""
    try:
        clf = Classifier(embeddings=embeddings, clf=LogisticRegression(max_iter=1000))
        metrics = clf.split_train_evaluate(X, Y, 0.8)
        return {
            'method': method_name,
            'accuracy': metrics['acc'],
            'f1_micro': metrics['micro'],
            'f1_macro': metrics['macro'],
            'success': True
        }
    except Exception as e:
        print(f"   ❌ {method_name} 评估失败: {e}")
        return {
            'method': method_name,
            'accuracy': 0.0,
            'f1_micro': 0.0,
            'f1_macro': 0.0,
            'success': False,
            'error': str(e)
        }

def run_baseline_struc2vec(G, X, Y, **kwargs):
    """运行基线Struc2Vec"""
    print("\\n🚀 训练基线 Struc2Vec...")
    start = time.time()
    
    try:
        model = Struc2Vec(
            G,
            walk_length=kwargs.get('walk_length', 40),
            num_walks=kwargs.get('num_walks', 8),
            workers=kwargs.get('workers', 1),
            verbose=0,
            opt1_reduce_len=True,
            opt2_reduce_sim_calc=True
        )
        model.train(
            embed_size=kwargs.get('embed_size', 64),
            window_size=5,
            workers=kwargs.get('workers', 1),
            iter=kwargs.get('iter', 3)
        )
        embeddings = model.get_embeddings()
        training_time = time.time() - start
        
        result = evaluate_method(embeddings, X, Y, "Baseline Struc2Vec")
        result['training_time'] = training_time
        
        print(f"   ✅ 完成: 准确率={result['accuracy']:.4f}, 时间={training_time:.2f}s")
        return result
        
    except Exception as e:
        print(f"   ❌ 基线方法失败: {e}")
        return {'method': 'Baseline Struc2Vec', 'success': False, 'training_time': 0, 'error': str(e)}

def run_advanced_fusion_method(G, X, Y, fusion_method, **kwargs):
    """运行高级融合方法"""
    print(f"\\n🚀 训练高级融合方法: {fusion_method}...")
    start = time.time()
    
    try:
        # 设置融合参数
        fusion_params = {}
        if fusion_method == 'attention':
            fusion_params = {'num_heads': 4, 'feature_dim': 64}
        elif fusion_method == 'pyramid':
            fusion_params = {'pyramid_levels': 3}
        elif fusion_method == 'spectral':
            fusion_params = {'num_eigenvectors': 10}
        elif fusion_method == 'community':
            fusion_params = {'resolution': 1.0}
        
        model = AdvancedFusionStruc2Vec(
            G,
            fusion_method=fusion_method,
            fusion_params=fusion_params,
            walk_length=kwargs.get('walk_length', 40),
            num_walks=kwargs.get('num_walks', 8),
            workers=kwargs.get('workers', 1),
            verbose=0,
            max_layer=kwargs.get('max_layer', 3),
            distance_method=kwargs.get('distance_method', 'frobenius'),
            use_orbit_selection=False
        )
        
        model.train(
            embed_size=kwargs.get('embed_size', 64),
            window_size=5,
            workers=kwargs.get('workers', 1),
            iter=kwargs.get('iter', 3)
        )
        embeddings = model.get_embeddings()
        training_time = time.time() - start
        
        method_name = f"Advanced Fusion ({fusion_method})"
        result = evaluate_method(embeddings, X, Y, method_name)
        result['training_time'] = training_time
        
        print(f"   ✅ 完成: 准确率={result['accuracy']:.4f}, 时间={training_time:.2f}s")
        return result
        
    except Exception as e:
        print(f"   ❌ 高级融合方法 {fusion_method} 失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            'method': f'Advanced Fusion ({fusion_method})', 
            'success': False, 
            'training_time': 0, 
            'error': str(e)
        }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='高级融合方法比较')
    parser.add_argument('--dataset', default='usa-airports',
                       choices=['usa-airports', 'wiki', 'lastfm'])
    parser.add_argument('--fusion-methods', default='attention,spectral,community,ensemble',
                       help='fusion methods to test (comma-separated)')
    parser.add_argument('--include-baseline', action='store_true', default=True,
                       help='include baseline Struc2Vec for comparison')
    parser.add_argument('--num-walks', type=int, default=6)
    parser.add_argument('--walk-length', type=int, default=30)
    parser.add_argument('--embed-size', type=int, default=64)
    parser.add_argument('--iter', type=int, default=2)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--max-layer', type=int, default=3)
    parser.add_argument('--distance-method', default='frobenius',
                       choices=['frobenius', 'eigenvalue', 'trace'])
    parser.add_argument('--num-runs', type=int, default=1,
                       help='number of runs for averaging results (default: 1)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 高级融合方法比较 (Advanced Fusion Methods)")
    print("=" * 80)
    print(f"数据集: {args.dataset}")
    print(f"融合方法: {args.fusion_methods}")
    print(f"包含基线: {args.include_baseline}")
    if args.num_runs > 1:
        print(f"运行次数: {args.num_runs}")
    
    try:
        # 加载数据
        G, X, Y = load_dataset(args.dataset)
        
        # 运行算法
        fusion_methods = [m.strip() for m in args.fusion_methods.split(',')]
        
        common_params = {
            'walk_length': args.walk_length,
            'num_walks': args.num_walks,
            'embed_size': args.embed_size,
            'iter': args.iter,
            'workers': args.workers,
            'max_layer': args.max_layer,
            'distance_method': args.distance_method
        }
        
        # 多次运行实验
        all_runs_results = []
        for run_id in range(1, args.num_runs + 1):
            run_results = run_single_experiment(G, X, Y, fusion_methods, common_params, args.include_baseline, run_id)
            all_runs_results.append(run_results)
        
        # 计算平均结果
        final_results = calculate_average_results(all_runs_results, args.num_runs)
        
        # 打印结果
        print_results(final_results, args.num_runs)
        
        print("\\n✅ 高级融合方法比较完成!")
        
    except Exception as e:
        print(f"\\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

def run_single_experiment(G, X, Y, fusion_methods, common_params, include_baseline=True, run_id=1):
    """运行单次实验"""
    print(f"\n🔄 第 {run_id} 次运行...")
    results = []
    
    # 基线方法
    if include_baseline:
        baseline_result = run_baseline_struc2vec(G, X, Y, **common_params)
        results.append(baseline_result)
    
    # 高级融合方法
    for fusion_method in fusion_methods:
        if fusion_method in ['attention', 'pyramid', 'spectral', 'community', 'ensemble']:
            result = run_advanced_fusion_method(G, X, Y, fusion_method, **common_params)
            results.append(result)
        else:
            print(f"⚠️  跳过未知融合方法: {fusion_method}")
    
    return results

def calculate_average_results(all_runs_results, num_runs):
    """计算多次运行的平均结果"""
    if num_runs == 1:
        return all_runs_results[0]
    
    print(f"\n📊 计算 {num_runs} 次运行的平均结果...")
    
    # 按方法名分组结果
    method_results = {}
    for run_results in all_runs_results:
        for result in run_results:
            method_name = result['method']
            if method_name not in method_results:
                method_results[method_name] = []
            method_results[method_name].append(result)
    
    # 计算每个方法的平均值和标准差
    averaged_results = []
    for method_name, results in method_results.items():
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            # 如果没有成功的结果，返回失败结果
            averaged_results.append({
                'method': method_name,
                'success': False,
                'accuracy': 0.0,
                'f1_micro': 0.0,
                'f1_macro': 0.0,
                'training_time': 0.0,
                'error': f"所有 {len(results)} 次运行都失败"
            })
        else:
            # 计算平均值和标准差
            accuracies = [r['accuracy'] for r in successful_results]
            f1_micros = [r['f1_micro'] for r in successful_results]
            f1_macros = [r['f1_macro'] for r in successful_results]
            training_times = [r['training_time'] for r in successful_results]
            
            averaged_results.append({
                'method': method_name,
                'success': True,
                'accuracy': np.mean(accuracies),
                'f1_micro': np.mean(f1_micros),
                'f1_macro': np.mean(f1_macros),
                'training_time': np.mean(training_times),
                'accuracy_std': np.std(accuracies),
                'f1_micro_std': np.std(f1_micros),
                'f1_macro_std': np.std(f1_macros),
                'training_time_std': np.std(training_times),
                'num_successful': len(successful_results),
                'num_total': len(results)
            })
    
    return averaged_results

def print_results(results, num_runs):
    """打印结果的函数"""
    # 打印结果
    print("\\n" + "=" * 80)
    if num_runs > 1:
        print(f"📊 高级融合方法比较结果 ({num_runs} 次运行平均值)")
    else:
        print("📊 高级融合方法比较结果")
    print("=" * 80)
    
    if num_runs > 1:
        print(f"\\n{'方法':<30} {'成功':<6} {'准确率':<12} {'F1-Micro':<12} {'F1-Macro':<12} {'时间(s)':<12}")
        print("-" * 100)
    else:
        print(f"\\n{'方法':<30} {'成功':<6} {'准确率':<10} {'F1-Micro':<10} {'F1-Macro':<10} {'时间(s)':<10}")
        print("-" * 90)
    
    successful_results = []
    for result in results:
        status = "✅" if result.get('success', False) else "❌"
        accuracy = result.get('accuracy', 0)
        f1_micro = result.get('f1_micro', 0)
        f1_macro = result.get('f1_macro', 0)
        time_taken = result.get('training_time', 0)
        
        if num_runs > 1 and result.get('success', False):
            # 显示均值±标准差
            acc_std = result.get('accuracy_std', 0)
            f1_micro_std = result.get('f1_micro_std', 0)
            f1_macro_std = result.get('f1_macro_std', 0)
            time_std = result.get('training_time_std', 0)
            success_rate = f"{result.get('num_successful', 0)}/{result.get('num_total', 0)}"
            
            print(f"{result['method']:<30} {success_rate:<6} {accuracy:.3f}±{acc_std:.3f} {f1_micro:.3f}±{f1_micro_std:.3f} {f1_macro:.3f}±{f1_macro_std:.3f} {time_taken:.1f}±{time_std:.1f}")
        else:
            print(f"{result['method']:<30} {status:<6} {accuracy:<10.4f} {f1_micro:<10.4f} {f1_macro:<10.4f} {time_taken:<10.2f}")
        
        if result.get('success', False):
            successful_results.append(result)
        elif 'error' in result:
            print(f"      错误: {result['error'][:60]}...")
    
    # 分析结果
    if successful_results:
        print("\\n" + "=" * 80)
        print("📈 性能分析")
        print("=" * 80)
        
        # 找到最佳方法
        best_result = max(successful_results, key=lambda x: x['accuracy'])
        print(f"\\n🏆 最佳方法: {best_result['method']}")
        print(f"🏆 最佳准确率: {best_result['accuracy']:.4f}")
        print(f"🏆 最佳F1-Micro: {best_result['f1_micro']:.4f}")
        
        # 与基线比较
        baseline_results = [r for r in successful_results if 'Baseline' in r['method']]
        if baseline_results:
            baseline = baseline_results[0]
            print(f"\\n📊 相对基线 Struc2Vec 的改进:")
            
            for result in successful_results:
                if 'Baseline' not in result['method']:
                    if baseline['accuracy'] > 0:
                        acc_improvement = (result['accuracy'] - baseline['accuracy']) / baseline['accuracy'] * 100
                        time_ratio = result['training_time'] / baseline['training_time'] if baseline['training_time'] > 0 else 0
                        
                        emoji = "📈" if acc_improvement > 0 else "📉"
                        print(f"   {emoji} {result['method']}: 准确率 {acc_improvement:+.1f}%, 时间比 {time_ratio:.1f}x")
        
        # 效率分析
        print(f"\\n⚡ 效率分析:")
        for result in successful_results:
            efficiency_score = result['accuracy'] / (result['training_time'] + 1e-6)
            print(f"   {result['method']}: 效率分数 {efficiency_score:.4f} (准确率/时间)")
        
        # 推荐
        print(f"\\n💡 推荐:")
        if best_result['accuracy'] > 0.7:
            print(f"   🎯 {best_result['method']} 表现优秀，推荐使用")
        elif best_result['accuracy'] > 0.5:
            print(f"   ⚠️  {best_result['method']} 表现中等，可考虑参数调优")
        else:
            print(f"   ❌ 所有方法表现较差，建议:")
            print(f"      - 检查数据质量")
            print(f"      - 尝试其他数据集") 
            print(f"      - 调整算法参数")
            
        # 特殊方法推荐
        attention_results = [r for r in successful_results if 'attention' in r['method'].lower()]
        if attention_results and attention_results[0]['accuracy'] > 0.6:
            print(f"   🤖 注意力机制表现良好，适合复杂图结构")
            
        spectral_results = [r for r in successful_results if 'spectral' in r['method'].lower()]
        if spectral_results and spectral_results[0]['accuracy'] > 0.6:
            print(f"   🌊 谱方法表现良好，适合具有明显社区结构的图")
            
        ensemble_results = [r for r in successful_results if 'ensemble' in r['method'].lower()]
        if ensemble_results and ensemble_results[0]['accuracy'] > 0.6:
            print(f"   🎭 集成方法表现良好，提供了稳定的性能")

if __name__ == "__main__":
    exit(main())
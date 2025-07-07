#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全的高级融合方法比较脚本
严格防止数据泄露的版本
"""
import sys
import os
import time
import argparse
import networkx as nx
from pathlib import Path

# 添加父项目路径
parent_path = Path(__file__).parent.parent
sys.path.append(str(parent_path))

from GraphEmbedding.ge.classify import read_node_label
from algorithms.traditional.struc2vec import Struc2Vec
from src.algorithms.advanced_fusion_methods import AdvancedFusionStruc2Vec
from safe_evaluation import safe_evaluate_method, DataLeakageChecker

def load_dataset(dataset_name):
    """加载数据集"""
    data_base = parent_path / 'data'
    
    datasets = {
        'brazil-airports': {
            'graph': data_base / 'flight/brazil-airports.edgelist',
            'labels': data_base / 'flight/labels-brazil-airports.txt'
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

def run_safe_baseline_struc2vec(G, X, Y, random_seed=42, **kwargs):
    """安全运行基线Struc2Vec"""
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
        
        # 验证实验设置
        print("🔍 验证基线方法实验设置...")
        DataLeakageChecker.validate_experimental_setup(embeddings, X, Y, train_ratio=0.8)
        
        # 安全评估
        result = safe_evaluate_method(embeddings, X, Y, "Baseline Struc2Vec", random_seed)
        result['training_time'] = training_time
        
        print(f"   ✅ 完成: 准确率={result['accuracy']:.4f}, 时间={training_time:.2f}s")
        return result
        
    except Exception as e:
        print(f"   ❌ 基线方法失败: {e}")
        return {'method': 'Baseline Struc2Vec', 'success': False, 'training_time': 0, 'error': str(e)}

def run_safe_advanced_fusion_method(G, X, Y, fusion_method, random_seed=42, **kwargs):
    """安全运行高级融合方法"""
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
        
        # 验证实验设置
        print(f"🔍 验证 {fusion_method} 方法实验设置...")
        DataLeakageChecker.validate_experimental_setup(embeddings, X, Y, train_ratio=0.8)
        
        # 安全评估
        method_name = f"Advanced Fusion ({fusion_method})"
        result = safe_evaluate_method(embeddings, X, Y, method_name, random_seed)
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
    parser = argparse.ArgumentParser(description='安全的高级融合方法比较')
    parser.add_argument('--dataset', default='brazil-airports', 
                       choices=['brazil-airports', 'wiki', 'lastfm'])
    parser.add_argument('--fusion-methods', default='attention,spectral',
                       help='fusion methods to test (comma-separated)')
    parser.add_argument('--include-baseline', action='store_true', default=True,
                       help='include baseline Struc2Vec for comparison')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='random seed for reproducible results')
    parser.add_argument('--num-walks', type=int, default=4)
    parser.add_argument('--walk-length', type=int, default=20)
    parser.add_argument('--embed-size', type=int, default=64)
    parser.add_argument('--iter', type=int, default=2)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--max-layer', type=int, default=3)
    parser.add_argument('--distance-method', default='frobenius',
                       choices=['frobenius', 'eigenvalue', 'trace'])
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🛡️  安全的高级融合方法比较 (Data Leakage Prevention)")
    print("=" * 80)
    print(f"数据集: {args.dataset}")
    print(f"融合方法: {args.fusion_methods}")
    print(f"包含基线: {args.include_baseline}")
    print(f"随机种子: {args.random_seed} (确保可重复性)")
    
    try:
        # 加载数据
        G, X, Y = load_dataset(args.dataset)
        
        # 数据泄露预检查
        print("\\n🔍 数据泄露预检查...")
        print(f"   图节点数: {G.number_of_nodes()}")
        print(f"   标记节点数: {len(X)}")
        print(f"   标记节点在图中的比例: {len(X)/G.number_of_nodes():.1%}")
        
        # 检查标记节点是否都在图中
        graph_nodes = set(G.nodes())
        label_nodes = set(X)
        missing_nodes = label_nodes - graph_nodes
        if missing_nodes:
            print(f"⚠️  警告: {len(missing_nodes)} 个标记节点不在图中")
        else:
            print("✅ 所有标记节点都在图中")
        
        # 运行算法
        fusion_methods = [m.strip() for m in args.fusion_methods.split(',')]
        results = []
        
        common_params = {
            'walk_length': args.walk_length,
            'num_walks': args.num_walks,
            'embed_size': args.embed_size,
            'iter': args.iter,
            'workers': args.workers,
            'max_layer': args.max_layer,
            'distance_method': args.distance_method
        }
        
        # 基线方法
        if args.include_baseline:
            baseline_result = run_safe_baseline_struc2vec(
                G, X, Y, random_seed=args.random_seed, **common_params
            )
            results.append(baseline_result)
        
        # 高级融合方法
        for fusion_method in fusion_methods:
            if fusion_method in ['attention', 'pyramid', 'spectral', 'community', 'ensemble']:
                result = run_safe_advanced_fusion_method(
                    G, X, Y, fusion_method, random_seed=args.random_seed, **common_params
                )
                results.append(result)
            else:
                print(f"⚠️  跳过未知融合方法: {fusion_method}")
        
        # 打印结果
        print("\\n" + "=" * 80)
        print("📊 安全比较结果 (无数据泄露)")
        print("=" * 80)
        
        print(f"\\n{'方法':<30} {'成功':<6} {'准确率':<10} {'F1-Micro':<10} {'F1-Macro':<10} {'训练集':<8} {'测试集':<8} {'时间(s)':<10}")
        print("-" * 110)
        
        successful_results = []
        for result in results:
            status = "✅" if result.get('success', False) else "❌"
            accuracy = result.get('accuracy', 0)
            f1_micro = result.get('f1_micro', 0)
            f1_macro = result.get('f1_macro', 0)
            train_size = result.get('train_size', 0)
            test_size = result.get('test_size', 0)
            time_taken = result.get('training_time', 0)
            
            print(f"{result['method']:<30} {status:<6} {accuracy:<10.4f} {f1_micro:<10.4f} {f1_macro:<10.4f} {train_size:<8} {test_size:<8} {time_taken:<10.2f}")
            
            if result.get('success', False):
                successful_results.append(result)
            elif 'error' in result:
                print(f"      错误: {result['error'][:50]}...")
        
        # 数据完整性最终检查
        print("\\n🔍 最终数据完整性检查:")
        for result in successful_results:
            train_size = result.get('train_size', 0)
            test_size = result.get('test_size', 0)
            total_eval = train_size + test_size
            print(f"   {result['method']}: 训练集={train_size}, 测试集={test_size}, 总计={total_eval}, 期望={len(X)}")
            
            if total_eval != len(X):
                print(f"   ⚠️  数据不一致: {result['method']}")
        
        # 性能分析
        if successful_results:
            print("\\n" + "=" * 80)
            print("📈 性能分析 (可信结果)")
            print("=" * 80)
            
            # 找到最佳方法
            best_result = max(successful_results, key=lambda x: x['accuracy'])
            print(f"\\n🏆 最佳方法: {best_result['method']}")
            print(f"🏆 最佳准确率: {best_result['accuracy']:.4f}")
            print(f"🏆 训练集大小: {best_result.get('train_size', 'N/A')}")
            print(f"🏆 测试集大小: {best_result.get('test_size', 'N/A')}")
            
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
            
            # 可重复性报告
            print(f"\\n🔬 可重复性报告:")
            print(f"   随机种子: {args.random_seed}")
            print(f"   所有结果基于相同的训练-测试分割")
            print(f"   严格防止数据泄露")
            print(f"   实验参数: walks={args.num_walks}, length={args.walk_length}, iter={args.iter}")
        
        print("\\n✅ 安全比较完成! 无数据泄露风险")
        
    except Exception as e:
        print(f"\\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
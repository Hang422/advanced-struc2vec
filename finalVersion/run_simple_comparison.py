#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的统一比较脚本
"""
import sys
import os
import time
import argparse
import networkx as nx
from pathlib import Path
from sklearn.linear_model import LogisticRegression

# 添加父项目路径
parent_path = Path(__file__).parent.parent
sys.path.append(str(parent_path))

# 应用 Struc2Vec 除零警告修复补丁
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from src.utils import struc2vec_patch
except:
    pass  # 补丁是可选的

from GraphEmbedding.ge.classify import read_node_label, Classifier
from algorithms.traditional.struc2vec import Struc2Vec
from algorithms.graphlet_based.compute_edges_improved import generate_improved_structural_distance

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
            'success': False
        }

def run_original_struc2vec(G, X, Y, **kwargs):
    """运行原始 Struc2Vec"""
    print("\\n🚀 训练原始 Struc2Vec...")
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
        
        result = evaluate_method(embeddings, X, Y, "Original Struc2Vec")
        result['training_time'] = training_time
        
        print(f"   ✅ 完成: 准确率={result['accuracy']:.4f}, 时间={training_time:.2f}s")
        return result
        
    except Exception as e:
        print(f"   ❌ 原始方法失败: {e}")
        return {'method': 'Original Struc2Vec', 'success': False, 'training_time': 0}

def run_graphlet_struc2vec(G, X, Y, **kwargs):
    """运行 Graphlet 增强 Struc2Vec"""
    print("\\n🚀 训练 Graphlet 增强 Struc2Vec...")
    start = time.time()
    
    try:
        # 生成距离文件
        output_dir = Path(__file__).parent / 'output' / 'distances'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存临时图文件
        temp_dir = Path(__file__).parent / 'temp'
        temp_dir.mkdir(parents=True, exist_ok=True)
        graph_file = temp_dir / f'temp_graph_{id(G)}.edgelist'
        nx.write_edgelist(G, str(graph_file), data=False)
        
        distance_file = output_dir / f'graphlet_distances_{id(G)}.pkl'
        
        # 生成 Graphlet 距离
        generate_improved_structural_distance(
            str(graph_file),
            str(distance_file),
            max_layer=kwargs.get('max_layer', 3),
            distance_method=kwargs.get('distance_method', 'frobenius'),
            use_orbit_selection=False
        )
        
        # 训练模型
        model = Struc2Vec(
            G,
            walk_length=kwargs.get('walk_length', 40),
            num_walks=kwargs.get('num_walks', 8),
            workers=kwargs.get('workers', 1),
            verbose=0,
            opt1_reduce_len=True,
            opt2_reduce_sim_calc=True,
            structural_dist_file=str(distance_file)
        )
        model.train(
            embed_size=kwargs.get('embed_size', 64),
            window_size=5,
            workers=kwargs.get('workers', 1),
            iter=kwargs.get('iter', 3)
        )
        embeddings = model.get_embeddings()
        training_time = time.time() - start
        
        result = evaluate_method(embeddings, X, Y, "Graphlet Struc2Vec")
        result['training_time'] = training_time
        
        print(f"   ✅ 完成: 准确率={result['accuracy']:.4f}, 时间={training_time:.2f}s")
        
        # 清理临时文件
        if graph_file.exists():
            graph_file.unlink()
        
        return result
        
    except Exception as e:
        print(f"   ❌ Graphlet 方法失败: {e}")
        return {'method': 'Graphlet Struc2Vec', 'success': False, 'training_time': 0}

def run_fusion_struc2vec(G, X, Y, alpha=0.5, **kwargs):
    """运行融合 Struc2Vec"""
    print(f"\\n🚀 训练融合 Struc2Vec (α={alpha})...")
    start = time.time()
    
    try:
        import pickle
        
        output_dir = Path(__file__).parent / 'output' / 'distances'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成原始距离
        temp_model = Struc2Vec(G, walk_length=10, num_walks=1, workers=1, verbose=0)
        import shutil
        time.sleep(1)
        original_dist_temp = temp_model.temp_path + "/structural_dist.pkl"
        original_dist_file = output_dir / f'original_distances_{id(G)}.pkl'
        
        if Path(original_dist_temp).exists():
            shutil.copy(original_dist_temp, original_dist_file)
            shutil.rmtree(temp_model.temp_path)
        else:
            raise FileNotFoundError("无法生成原始距离文件")
        
        # 生成 Graphlet 距离（重用之前的）
        graphlet_dist_file = output_dir / f'graphlet_distances_{id(G)}.pkl'
        if not graphlet_dist_file.exists():
            # 如果不存在，先生成
            temp_dir = Path(__file__).parent / 'temp'
            temp_dir.mkdir(parents=True, exist_ok=True)
            graph_file = temp_dir / f'temp_graph_{id(G)}.edgelist'
            nx.write_edgelist(G, str(graph_file), data=False)
            
            generate_improved_structural_distance(
                str(graph_file),
                str(graphlet_dist_file),
                max_layer=3,
                distance_method='frobenius',
                use_orbit_selection=False
            )
            
            if graph_file.exists():
                graph_file.unlink()
        
        # 融合距离
        with open(original_dist_file, 'rb') as f:
            dist1 = pickle.load(f)
        with open(graphlet_dist_file, 'rb') as f:
            dist2 = pickle.load(f)
        
        fused = {}
        for pair in set(dist1.keys()).intersection(dist2.keys()):
            fused[pair] = {}
            layers1 = dist1[pair]
            layers2 = dist2[pair]
            for layer in set(layers1.keys()).intersection(layers2.keys()):
                fused[pair][layer] = alpha * layers1[layer] + (1 - alpha) * layers2[layer]
        
        fused_file = output_dir / f'fused_distances_{alpha}_{id(G)}.pkl'
        with open(fused_file, 'wb') as f:
            pickle.dump(fused, f)
        
        # 训练模型
        model = Struc2Vec(
            G,
            walk_length=kwargs.get('walk_length', 40),
            num_walks=kwargs.get('num_walks', 8),
            workers=kwargs.get('workers', 1),
            verbose=0,
            opt1_reduce_len=True,
            opt2_reduce_sim_calc=True,
            structural_dist_file=str(fused_file)
        )
        model.train(
            embed_size=kwargs.get('embed_size', 64),
            window_size=5,
            workers=kwargs.get('workers', 1),
            iter=kwargs.get('iter', 3)
        )
        embeddings = model.get_embeddings()
        training_time = time.time() - start
        
        result = evaluate_method(embeddings, X, Y, f"Fusion Struc2Vec (α={alpha})")
        result['training_time'] = training_time
        
        print(f"   ✅ 完成: 准确率={result['accuracy']:.4f}, 时间={training_time:.2f}s")
        return result
        
    except Exception as e:
        print(f"   ❌ 融合方法失败: {e}")
        return {'method': f'Fusion Struc2Vec (α={alpha})', 'success': False, 'training_time': 0}

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Struc2Vec 算法统一比较')
    parser.add_argument('--dataset', default='brazil-airports', 
                       choices=['brazil-airports', 'wiki', 'lastfm'])
    parser.add_argument('--methods', default='original,graphlet,fusion',
                       help='methods to compare (comma-separated)')
    parser.add_argument('--fusion-alpha', type=float, default=0.5)
    parser.add_argument('--num-walks', type=int, default=8)
    parser.add_argument('--walk-length', type=int, default=40)
    parser.add_argument('--embed-size', type=int, default=64)
    parser.add_argument('--iter', type=int, default=3)
    parser.add_argument('--workers', type=int, default=1)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 Struc2Vec 算法统一比较 (Final Version)")
    print("=" * 80)
    print(f"数据集: {args.dataset}")
    print(f"比较方法: {args.methods}")
    
    try:
        # 加载数据
        G, X, Y = load_dataset(args.dataset)
        
        # 运行算法
        methods = [m.strip() for m in args.methods.split(',')]
        results = []
        
        common_params = {
            'walk_length': args.walk_length,
            'num_walks': args.num_walks,
            'embed_size': args.embed_size,
            'iter': args.iter,
            'workers': args.workers
        }
        
        if 'original' in methods:
            result = run_original_struc2vec(G, X, Y, **common_params)
            results.append(result)
        
        if 'graphlet' in methods:
            result = run_graphlet_struc2vec(G, X, Y, **common_params)
            results.append(result)
        
        if 'fusion' in methods:
            result = run_fusion_struc2vec(G, X, Y, alpha=args.fusion_alpha, **common_params)
            results.append(result)
        
        # 打印结果
        print("\\n" + "=" * 80)
        print("📊 比较结果")
        print("=" * 80)
        
        print(f"\\n{'方法':<25} {'成功':<6} {'准确率':<10} {'F1-Micro':<10} {'F1-Macro':<10} {'时间(s)':<10}")
        print("-" * 80)
        
        successful_results = []
        for result in results:
            status = "✅" if result.get('success', False) else "❌"
            accuracy = result.get('accuracy', 0)
            f1_micro = result.get('f1_micro', 0)
            f1_macro = result.get('f1_macro', 0)
            time_taken = result.get('training_time', 0)
            
            print(f"{result['method']:<25} {status:<6} {accuracy:<10.4f} {f1_micro:<10.4f} {f1_macro:<10.4f} {time_taken:<10.2f}")
            
            if result.get('success', False):
                successful_results.append(result)
        
        # 分析结果
        if successful_results:
            best_result = max(successful_results, key=lambda x: x['accuracy'])
            print(f"\\n🏆 最佳方法: {best_result['method']}")
            print(f"🏆 最佳准确率: {best_result['accuracy']:.4f}")
            
            if len(successful_results) > 1:
                baseline = successful_results[0]
                print(f"\\n📈 相对 {baseline['method']} 的改进:")
                for result in successful_results[1:]:
                    if baseline['accuracy'] > 0:
                        improvement = (result['accuracy'] - baseline['accuracy']) / baseline['accuracy'] * 100
                        emoji = "📈" if improvement > 0 else "📉"
                        print(f"   {emoji} {result['method']}: {improvement:+.1f}%")
        
        print("\\n✅ 比较完成!")
        
    except Exception as e:
        print(f"\\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
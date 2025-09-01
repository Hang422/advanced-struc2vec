#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Fusion Methods Comparison Script
Test and compare multiple advanced feature fusion techniques
"""
import sys
import os
import time
import argparse
import numpy as np
import networkx as nx
from pathlib import Path
from sklearn.linear_model import LogisticRegression

# Add parent project path
parent_path = Path(__file__).parent.parent
sys.path.append(str(parent_path))

# Apply Struc2Vec divide-by-zero warning fix patch
try:
    from src.utils import struc2vec_patch
except:
    pass  # Patch is optional

from libs.GraphEmbedding.ge.classify import read_node_label, Classifier
from libs.GraphEmbedding.ge.models.struc2vec import Struc2Vec
from src.algorithms.advanced_fusion_methods import AdvancedFusionStruc2Vec, PureGraphletStruc2Vec

def load_dataset(dataset_name):
    """Load dataset"""
    data_base = parent_path / 'data'
    
    datasets = {
        'usa-airports': {
            'graph': data_base / 'flight/usa-airports.edgelist',
            'labels': data_base / 'flight/labels-usa-airports.txt',
            'description': 'USA airports network'
        },
        'brazil-airports': {
            'graph': data_base / 'flight/brazil-airports.edgelist',
            'labels': data_base / 'flight/labels-brazil-airports.txt',
            'description': 'Brazil airports network'
        },
        'europe-airports': {
            'graph': data_base / 'flight/europe-airports.edgelist',
            'labels': data_base / 'flight/labels-europe-airports.txt',
            'description': 'Europe airports network'
        },
        'wiki': {
            'graph': data_base / 'wiki/Wiki_edgelist.txt',
            'labels': data_base / 'wiki/wiki_labels.txt',
            'description': 'Wikipedia network'
        },
        'lastfm': {
            'graph': data_base / 'lastfm_asia/lastfm_asia.edgelist',
            'labels': data_base / 'lastfm_asia/lastfm_asia_labels.txt',
            'description': 'LastFM Asia social network'
        }
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    info = datasets[dataset_name]
    
    print(f"📂 Loading dataset: {dataset_name}")
    print(f"   Description: {info['description']}")
    print(f"   Graph file: {info['graph']}")
    print(f"   Label file: {info['labels']}")
    
    # Load graph
    G = nx.read_edgelist(str(info['graph']), nodetype=str, create_using=nx.DiGraph())
    print(f"   Graph info: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Load labels
    X, Y = read_node_label(str(info['labels']), skip_head=True)
    print(f"   Label info: {len(X)} labeled nodes")
    
    return G, X, Y

def evaluate_method(embeddings, X, Y, method_name):
    """Evaluate single method"""
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
        print(f"   ❌ {method_name} evaluation failed: {e}")
        return {
            'method': method_name,
            'accuracy': 0.0,
            'f1_micro': 0.0,
            'f1_macro': 0.0,
            'success': False,
            'error': str(e)
        }

def run_baseline_struc2vec(G, X, Y, **kwargs):
    """Run baseline Struc2Vec"""
    print("\\n🚀 Training baseline Struc2Vec...")
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
        
        print(f"   ✅ Completed: accuracy={result['accuracy']:.4f}, time={training_time:.2f}s")
        return result
        
    except Exception as e:
        print(f"   ❌ Baseline method failed: {e}")
        return {'method': 'Baseline Struc2Vec', 'success': False, 'training_time': 0, 'error': str(e)}

def run_advanced_fusion_method(G, X, Y, fusion_method, **kwargs):
    """Run advanced fusion method"""
    print(f"\\n🚀 Training advanced fusion method: {fusion_method}...")
    start = time.time()
    
    try:
        # Set fusion parameters
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
        
        print(f"   ✅ Completed: accuracy={result['accuracy']:.4f}, time={training_time:.2f}s")
        return result
        
    except Exception as e:
        print(f"   ❌ Advanced fusion method {fusion_method} failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'method': f'Advanced Fusion ({fusion_method})', 
            'success': False, 
            'training_time': 0, 
            'error': str(e)
        }

def run_pure_graphlet_method(G, X, Y, **kwargs):
    """Run pure graphlet method"""
    print(f"\\n🚀 Training pure graphlet method...")
    start = time.time()
    
    try:
        model = PureGraphletStruc2Vec(
            G,
            walk_length=kwargs.get('walk_length', 40),
            num_walks=kwargs.get('num_walks', 8),
            workers=kwargs.get('workers', 1),
            verbose=0,
            k=kwargs.get('k', 5),
            max_layer=kwargs.get('max_layer', 3),
            distance_method=kwargs.get('distance_method', 'frobenius'),
            use_orbit_selection=kwargs.get('use_orbit_selection', False),
            top_k_orbits=kwargs.get('top_k_orbits', 40)
        )
        
        model.train(
            embed_size=kwargs.get('embed_size', 64),
            window_size=5,
            workers=kwargs.get('workers', 1),
            iter=kwargs.get('iter', 3)
        )
        embeddings = model.get_embeddings()
        training_time = time.time() - start
        
        method_name = "Pure Graphlet Struc2Vec"
        result = evaluate_method(embeddings, X, Y, method_name)
        result['training_time'] = training_time
        
        print(f"   ✅ Completed: accuracy={result['accuracy']:.4f}, time={training_time:.2f}s")
        return result
        
    except Exception as e:
        print(f"   ❌ Pure graphlet method failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'method': 'Pure Graphlet Struc2Vec', 
            'success': False, 
            'training_time': 0, 
            'error': str(e)
        }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Advanced Fusion Methods Comparison')
    parser.add_argument('--dataset', default='usa-airports',
                       choices=['usa-airports', 'brazil-airports', 'europe-airports', 'wiki', 'lastfm'])
    parser.add_argument('--fusion-methods', default='attention,pyramid,spectral,community,ensemble,pure-graphlet',
                       help='fusion methods to test (comma-separated, including pure-graphlet)')
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
    print("🚀 Advanced Fusion Methods Comparison")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Fusion methods: {args.fusion_methods}")
    print(f"Include baseline: {args.include_baseline}")
    if args.num_runs > 1:
        print(f"Number of runs: {args.num_runs}")
    
    try:
        # Load data
        G, X, Y = load_dataset(args.dataset)
        
        # Run algorithms
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
        
        # Run multiple experiments
        all_runs_results = []
        for run_id in range(1, args.num_runs + 1):
            run_results = run_single_experiment(G, X, Y, fusion_methods, common_params, args.include_baseline, run_id)
            all_runs_results.append(run_results)
        
        # Calculate average results
        final_results = calculate_average_results(all_runs_results, args.num_runs)
        
        # Print results
        print_results(final_results, args.num_runs)
        
        print("\\n✅ Advanced fusion methods comparison completed!")
        
    except Exception as e:
        print(f"\\n❌ Execution error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

def run_single_experiment(G, X, Y, fusion_methods, common_params, include_baseline=True, run_id=1):
    """Run single experiment"""
    print(f"\n🔄 Run #{run_id}...")
    results = []
    
    # Baseline method
    if include_baseline:
        baseline_result = run_baseline_struc2vec(G, X, Y, **common_params)
        results.append(baseline_result)
    
    # Advanced fusion methods and pure graphlet method
    for fusion_method in fusion_methods:
        if fusion_method in ['attention', 'pyramid', 'spectral', 'community', 'ensemble']:
            result = run_advanced_fusion_method(G, X, Y, fusion_method, **common_params)
            results.append(result)
        elif fusion_method == 'pure-graphlet':
            result = run_pure_graphlet_method(G, X, Y, **common_params)
            results.append(result)
        else:
            print(f"⚠️  Skipping unknown fusion method: {fusion_method}")
    
    return results

def calculate_average_results(all_runs_results, num_runs):
    """Calculate average results from multiple runs"""
    if num_runs == 1:
        return all_runs_results[0]
    
    print(f"\n📊 Calculating average results from {num_runs} runs...")
    
    # Group results by method name
    method_results = {}
    for run_results in all_runs_results:
        for result in run_results:
            method_name = result['method']
            if method_name not in method_results:
                method_results[method_name] = []
            method_results[method_name].append(result)
    
    # Calculate mean and standard deviation for each method
    averaged_results = []
    for method_name, results in method_results.items():
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            # If no successful results, return failure result
            averaged_results.append({
                'method': method_name,
                'success': False,
                'accuracy': 0.0,
                'f1_micro': 0.0,
                'f1_macro': 0.0,
                'training_time': 0.0,
                'error': f"All {len(results)} runs failed"
            })
        else:
            # Calculate mean and standard deviation
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
    """Function to print results"""
    # Print results
    print("\\n" + "=" * 80)
    if num_runs > 1:
        print(f"📊 Advanced Fusion Methods Comparison Results (Average of {num_runs} runs)")
    else:
        print("📊 Advanced Fusion Methods Comparison Results")
    print("=" * 80)
    
    if num_runs > 1:
        print(f"\\n{'Method':<30} {'Success':<6} {'Accuracy':<12} {'F1-Micro':<12} {'F1-Macro':<12} {'Time(s)':<12}")
        print("-" * 100)
    else:
        print(f"\\n{'Method':<30} {'Success':<6} {'Accuracy':<10} {'F1-Micro':<10} {'F1-Macro':<10} {'Time(s)':<10}")
        print("-" * 90)
    
    successful_results = []
    for result in results:
        status = "✅" if result.get('success', False) else "❌"
        accuracy = result.get('accuracy', 0)
        f1_micro = result.get('f1_micro', 0)
        f1_macro = result.get('f1_macro', 0)
        time_taken = result.get('training_time', 0)
        
        if num_runs > 1 and result.get('success', False):
            # Display mean ± standard deviation
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
            print(f"      Error: {result['error'][:60]}...")
    
    # Analyze results
    if successful_results:
        print("\\n" + "=" * 80)
        print("📈 Performance Analysis")
        print("=" * 80)
        
        # Find best method
        best_result = max(successful_results, key=lambda x: x['accuracy'])
        print(f"\\n🏆 Best method: {best_result['method']}")
        print(f"🏆 Best accuracy: {best_result['accuracy']:.4f}")
        print(f"🏆 Best F1-Micro: {best_result['f1_micro']:.4f}")
        
        # Compare with baseline
        baseline_results = [r for r in successful_results if 'Baseline' in r['method']]
        if baseline_results:
            baseline = baseline_results[0]
            print(f"\\n📊 Improvements relative to baseline Struc2Vec:")
            
            for result in successful_results:
                if 'Baseline' not in result['method']:
                    if baseline['accuracy'] > 0:
                        acc_improvement = (result['accuracy'] - baseline['accuracy']) / baseline['accuracy'] * 100
                        time_ratio = result['training_time'] / baseline['training_time'] if baseline['training_time'] > 0 else 0
                        
                        emoji = "📈" if acc_improvement > 0 else "📉"
                        print(f"   {emoji} {result['method']}: accuracy {acc_improvement:+.1f}%, time ratio {time_ratio:.1f}x")
        
        # Efficiency analysis
        print(f"\\n⚡ Efficiency Analysis:")
        for result in successful_results:
            efficiency_score = result['accuracy'] / (result['training_time'] + 1e-6)
            print(f"   {result['method']}: efficiency score {efficiency_score:.4f} (accuracy/time)")
        
        # Recommendations
        print(f"\\n💡 Recommendations:")
        if best_result['accuracy'] > 0.7:
            print(f"   🎯 {best_result['method']} performs excellently, recommended for use")
        elif best_result['accuracy'] > 0.5:
            print(f"   ⚠️  {best_result['method']} performs moderately, consider parameter tuning")
        else:
            print(f"   ❌ All methods perform poorly, suggestions:")
            print(f"      - Check data quality")
            print(f"      - Try other datasets") 
            print(f"      - Adjust algorithm parameters")
            
        # Special method recommendations
        attention_results = [r for r in successful_results if 'attention' in r['method'].lower()]
        if attention_results and attention_results[0]['accuracy'] > 0.6:
            print(f"   🤖 Attention mechanism performs well, suitable for complex graph structures")
            
        spectral_results = [r for r in successful_results if 'spectral' in r['method'].lower()]
        if spectral_results and spectral_results[0]['accuracy'] > 0.6:
            print(f"   🌊 Spectral method performs well, suitable for graphs with clear community structure")
            
        ensemble_results = [r for r in successful_results if 'ensemble' in r['method'].lower()]
        if ensemble_results and ensemble_results[0]['accuracy'] > 0.6:
            print(f"   🎭 Ensemble method performs well, provides stable performance")

if __name__ == "__main__":
    exit(main())
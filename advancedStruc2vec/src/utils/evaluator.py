#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation Tools
"""
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Add parent project path
parent_path = Path(__file__).parent.parent.parent.parent
sys.path.append(str(parent_path))

from GraphEmbedding.ge.classify import Classifier

class Evaluator:
    """Algorithm evaluator"""
    
    def __init__(self, **kwargs):
        """
        Initialize evaluator
        
        Args:
            **kwargs: Evaluation parameters
        """
        self.train_ratio = kwargs.get('train_ratio', 0.8)
        self.random_state = kwargs.get('random_state', 42)
        self.classifiers = kwargs.get('classifiers', ['logistic'])
        self.metrics = kwargs.get('metrics', ['accuracy', 'f1_micro', 'f1_macro'])
        
        # Classifier mapping
        self.classifier_map = {
            'logistic': LogisticRegression(max_iter=1000, random_state=self.random_state),
            'svm': SVC(kernel='rbf', random_state=self.random_state),
            'rf': RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        }
    
    def evaluate_single(self, embeddings: Dict[str, np.ndarray], 
                       X: List, Y: List, 
                       classifier: str = 'logistic') -> Dict[str, float]:
        """
        Evaluate embeddings using a single classifier
        
        Args:
            embeddings: Node embeddings dictionary
            X: Node list
            Y: Label list
            classifier: Classifier name
            
        Returns:
            Evaluation metrics dictionary
        """
        if classifier not in self.classifier_map:
            raise ValueError(f"Unknown classifier: {classifier}")
        
        clf_model = self.classifier_map[classifier]
        clf = Classifier(embeddings=embeddings, clf=clf_model)
        
        try:
            metrics = clf.split_train_evaluate(X, Y, self.train_ratio)
            return metrics
        except Exception as e:
            print(f"   ❌ Evaluation failed ({classifier}): {e}")
            return {'acc': 0.0, 'micro': 0.0, 'macro': 0.0}
    
    def evaluate_multiple(self, embeddings: Dict[str, np.ndarray], 
                         X: List, Y: List) -> Dict[str, Dict[str, float]]:
        """
        Evaluate embeddings using multiple classifiers
        
        Args:
            embeddings: Node embeddings dictionary
            X: Node list
            Y: Label list
            
        Returns:
            Dictionary of {classifier_name: metrics}
        """
        results = {}
        
        for classifier in self.classifiers:
            print(f"   📊 Evaluating {classifier}...")
            metrics = self.evaluate_single(embeddings, X, Y, classifier)
            results[classifier] = metrics
            
            if metrics['acc'] > 0:
                print(f"      Accuracy: {metrics['acc']:.4f}, F1-Micro: {metrics['micro']:.4f}")
        
        return results
    
    def evaluate_algorithm(self, algorithm, X: List, Y: List, 
                          method_name: str = None) -> Dict[str, Any]:
        """
        Complete evaluation of a single algorithm
        
        Args:
            algorithm: Algorithm instance
            X: Node list
            Y: Label list
            method_name: Method name
            
        Returns:
            Complete evaluation results
        """
        if method_name is None:
            method_name = getattr(algorithm, 'get_method_name', lambda: 'Unknown')()
        
        print(f"\n🔬 Evaluating {method_name}...")
        
        # Training timing
        start_time = time.time()
        
        try:
            # Train algorithm
            algorithm.train()
            embeddings = algorithm.get_embeddings()
            training_time = time.time() - start_time
            
            # Evaluate embeddings
            classifier_results = self.evaluate_multiple(embeddings, X, Y)
            
            # Summarize results
            result = {
                'method_name': method_name,
                'training_time': training_time,
                'embedding_size': len(embeddings),
                'embedding_dim': list(embeddings.values())[0].shape[0] if embeddings else 0,
                'classifier_results': classifier_results,
                'success': True
            }
            
            # Compute best metrics
            best_metrics = self._get_best_metrics(classifier_results)
            result.update(best_metrics)
            
            print(f"   ✅ Training time: {training_time:.2f}s, Best accuracy: {result['best_accuracy']:.4f}")
            
        except Exception as e:
            print(f"   ❌ Evaluation failed: {e}")
            result = {
                'method_name': method_name,
                'training_time': 0,
                'embedding_size': 0,
                'embedding_dim': 0,
                'classifier_results': {},
                'success': False,
                'error': str(e),
                'best_accuracy': 0.0,
                'best_f1_micro': 0.0,
                'best_f1_macro': 0.0
            }
        
        return result
    
    def _get_best_metrics(self, classifier_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Get best metrics from multiple classifier results
        
        Args:
            classifier_results: Classifier results dictionary
            
        Returns:
            Best metrics dictionary
        """
        if not classifier_results:
            return {'best_accuracy': 0.0, 'best_f1_micro': 0.0, 'best_f1_macro': 0.0}
        
        best_acc = max(results['acc'] for results in classifier_results.values())
        best_micro = max(results['micro'] for results in classifier_results.values())
        best_macro = max(results['macro'] for results in classifier_results.values())
        
        return {
            'best_accuracy': best_acc,
            'best_f1_micro': best_micro,
            'best_f1_macro': best_macro
        }
    
    def compare_algorithms(self, algorithms: List, X: List, Y: List) -> Dict[str, Any]:
        """
        Compare multiple algorithms
        
        Args:
            algorithms: Algorithm list
            X: Node list
            Y: Label list
            
        Returns:
            Comparison results dictionary
        """
        print("\n" + "=" * 80)
        print("Algorithm Comparison Evaluation")
        print("=" * 80)
        
        results = {}
        
        for i, algorithm in enumerate(algorithms):
            try:
                method_name = getattr(algorithm, 'get_method_name', lambda: f'Method_{i+1}')()
                result = self.evaluate_algorithm(algorithm, X, Y, method_name)
                results[method_name] = result
            except Exception as e:
                print(f"   ❌ Algorithm {i+1} evaluation failed: {e}")
                continue
        
        # Generate comparison report
        comparison_report = self._generate_comparison_report(results)
        
        return {
            'individual_results': results,
            'comparison_report': comparison_report
        }
    
    def _generate_comparison_report(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comparison report
        
        Args:
            results: Algorithm results dictionary
            
        Returns:
            Comparison report
        """
        successful_results = {k: v for k, v in results.items() if v['success']}
        
        if not successful_results:
            return {'error': 'No successful algorithm results'}
        
        # Find best method
        best_method = max(successful_results.keys(), 
                         key=lambda k: successful_results[k]['best_accuracy'])
        best_accuracy = successful_results[best_method]['best_accuracy']
        
        # Compute relative improvements
        baseline_method = list(successful_results.keys())[0]  # Use first as baseline
        baseline_accuracy = successful_results[baseline_method]['best_accuracy']
        
        improvements = {}
        for method, result in successful_results.items():
            if method != baseline_method:
                if baseline_accuracy > 0:
                    improvement = (result['best_accuracy'] - baseline_accuracy) / baseline_accuracy * 100
                else:
                    improvement = 0
                improvements[method] = improvement
        
        # Efficiency analysis
        efficiency_analysis = {}
        baseline_time = successful_results[baseline_method]['training_time']
        for method, result in successful_results.items():
            if result['training_time'] > 0:
                speedup = baseline_time / result['training_time']
            else:
                speedup = 0
            efficiency_analysis[method] = {
                'training_time': result['training_time'],
                'speedup': speedup
            }
        
        return {
            'total_methods': len(results),
            'successful_methods': len(successful_results),
            'best_method': best_method,
            'best_accuracy': best_accuracy,
            'baseline_method': baseline_method,
            'baseline_accuracy': baseline_accuracy,
            'improvements': improvements,
            'efficiency_analysis': efficiency_analysis
        }
    
    def print_comparison_report(self, comparison_data: Dict[str, Any]) -> None:
        """
        Print comparison report
        
        Args:
            comparison_data: Comparison data
        """
        report = comparison_data['comparison_report']
        results = comparison_data['individual_results']
        
        if 'error' in report:
            print(f"❌ {report['error']}")
            return
        
        print("\n" + "=" * 80)
        print("📊 Algorithm Comparison Report")
        print("=" * 80)
        
        # Results summary table
        print(f"\n{'Method':<25} {'Success':<6} {'Accuracy':<10} {'F1-Micro':<10} {'F1-Macro':<10} {'Time(s)':<10}")
        print("-" * 80)
        
        for method, result in results.items():
            status = "✅" if result['success'] else "❌"
            accuracy = result.get('best_accuracy', 0)
            f1_micro = result.get('best_f1_micro', 0)
            f1_macro = result.get('best_f1_macro', 0)
            time_taken = result.get('training_time', 0)
            
            print(f"{method:<25} {status:<6} {accuracy:<10.4f} {f1_micro:<10.4f} {f1_macro:<10.4f} {time_taken:<10.2f}")
        
        # Best method
        print(f"\n🏆 Best method: {report['best_method']}")
        print(f"🏆 Best accuracy: {report['best_accuracy']:.4f}")
        
        # Relative improvements
        if report['improvements']:
            print(f"\n📈 Improvements relative to {report['baseline_method']}:")
            for method, improvement in report['improvements'].items():
                emoji = "📈" if improvement > 0 else "📉"
                print(f"   {emoji} {method}: {improvement:+.1f}%")
        
        # Efficiency analysis
        print(f"\n⚡ Efficiency Analysis:")
        for method, efficiency in report['efficiency_analysis'].items():
            print(f"   {method}: {efficiency['training_time']:.2f}s ({efficiency['speedup']:.1f}x)")

if __name__ == "__main__":
    # Test evaluator
    evaluator = Evaluator()
    print("Evaluator initialized successfully")
    print(f"Supported classifiers: {list(evaluator.classifier_map.keys())}")
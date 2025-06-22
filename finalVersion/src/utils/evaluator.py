#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估工具
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

# 添加父项目路径
parent_path = Path(__file__).parent.parent.parent.parent
sys.path.append(str(parent_path))

from GraphEmbedding.ge.classify import Classifier

class Evaluator:
    """算法评估器"""
    
    def __init__(self, **kwargs):
        """
        初始化评估器
        
        Args:
            **kwargs: 评估参数
        """
        self.train_ratio = kwargs.get('train_ratio', 0.8)
        self.random_state = kwargs.get('random_state', 42)
        self.classifiers = kwargs.get('classifiers', ['logistic'])
        self.metrics = kwargs.get('metrics', ['accuracy', 'f1_micro', 'f1_macro'])
        
        # 分类器映射
        self.classifier_map = {
            'logistic': LogisticRegression(max_iter=1000, random_state=self.random_state),
            'svm': SVC(kernel='rbf', random_state=self.random_state),
            'rf': RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        }
    
    def evaluate_single(self, embeddings: Dict[str, np.ndarray], 
                       X: List, Y: List, 
                       classifier: str = 'logistic') -> Dict[str, float]:
        """
        使用单个分类器评估嵌入
        
        Args:
            embeddings: 节点嵌入字典
            X: 节点列表
            Y: 标签列表
            classifier: 分类器名称
            
        Returns:
            评估指标字典
        """
        if classifier not in self.classifier_map:
            raise ValueError(f"未知分类器: {classifier}")
        
        clf_model = self.classifier_map[classifier]
        clf = Classifier(embeddings=embeddings, clf=clf_model)
        
        try:
            metrics = clf.split_train_evaluate(X, Y, self.train_ratio)
            return metrics
        except Exception as e:
            print(f"   ❌ 评估失败 ({classifier}): {e}")
            return {'acc': 0.0, 'micro': 0.0, 'macro': 0.0}
    
    def evaluate_multiple(self, embeddings: Dict[str, np.ndarray], 
                         X: List, Y: List) -> Dict[str, Dict[str, float]]:\n        \"\"\"\n        使用多个分类器评估嵌入\n        \n        Args:\n            embeddings: 节点嵌入字典\n            X: 节点列表\n            Y: 标签列表\n            \n        Returns:\n            {classifier_name: metrics} 的字典\n        \"\"\"\n        results = {}\n        \n        for classifier in self.classifiers:\n            print(f\"   📊 评估 {classifier}...\")\n            metrics = self.evaluate_single(embeddings, X, Y, classifier)\n            results[classifier] = metrics\n            \n            if metrics['acc'] > 0:\n                print(f\"      准确率: {metrics['acc']:.4f}, F1-Micro: {metrics['micro']:.4f}\")\n        \n        return results\n    \n    def evaluate_algorithm(self, algorithm, X: List, Y: List, \n                          method_name: str = None) -> Dict[str, Any]:\n        \"\"\"\n        完整评估单个算法\n        \n        Args:\n            algorithm: 算法实例\n            X: 节点列表\n            Y: 标签列表\n            method_name: 方法名称\n            \n        Returns:\n            完整评估结果\n        \"\"\"\n        if method_name is None:\n            method_name = getattr(algorithm, 'get_method_name', lambda: 'Unknown')()\n        \n        print(f\"\\n🔬 评估 {method_name}...\")\n        \n        # 训练计时\n        start_time = time.time()\n        \n        try:\n            # 训练算法\n            algorithm.train()\n            embeddings = algorithm.get_embeddings()\n            training_time = time.time() - start_time\n            \n            # 评估嵌入\n            classifier_results = self.evaluate_multiple(embeddings, X, Y)\n            \n            # 汇总结果\n            result = {\n                'method_name': method_name,\n                'training_time': training_time,\n                'embedding_size': len(embeddings),\n                'embedding_dim': list(embeddings.values())[0].shape[0] if embeddings else 0,\n                'classifier_results': classifier_results,\n                'success': True\n            }\n            \n            # 计算最佳指标\n            best_metrics = self._get_best_metrics(classifier_results)\n            result.update(best_metrics)\n            \n            print(f\"   ✅ 训练时间: {training_time:.2f}s, 最佳准确率: {result['best_accuracy']:.4f}\")\n            \n        except Exception as e:\n            print(f\"   ❌ 评估失败: {e}\")\n            result = {\n                'method_name': method_name,\n                'training_time': 0,\n                'embedding_size': 0,\n                'embedding_dim': 0,\n                'classifier_results': {},\n                'success': False,\n                'error': str(e),\n                'best_accuracy': 0.0,\n                'best_f1_micro': 0.0,\n                'best_f1_macro': 0.0\n            }\n        \n        return result\n    \n    def _get_best_metrics(self, classifier_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:\n        \"\"\"\n        从多个分类器结果中获取最佳指标\n        \n        Args:\n            classifier_results: 分类器结果字典\n            \n        Returns:\n            最佳指标字典\n        \"\"\"\n        if not classifier_results:\n            return {'best_accuracy': 0.0, 'best_f1_micro': 0.0, 'best_f1_macro': 0.0}\n        \n        best_acc = max(results['acc'] for results in classifier_results.values())\n        best_micro = max(results['micro'] for results in classifier_results.values())\n        best_macro = max(results['macro'] for results in classifier_results.values())\n        \n        return {\n            'best_accuracy': best_acc,\n            'best_f1_micro': best_micro,\n            'best_f1_macro': best_macro\n        }\n    \n    def compare_algorithms(self, algorithms: List, X: List, Y: List) -> Dict[str, Any]:\n        \"\"\"\n        比较多个算法\n        \n        Args:\n            algorithms: 算法列表\n            X: 节点列表\n            Y: 标签列表\n            \n        Returns:\n            比较结果字典\n        \"\"\"\n        print(\"\\n\" + \"=\" * 80)\n        print(\"算法比较评估\")\n        print(\"=\" * 80)\n        \n        results = {}\n        \n        for i, algorithm in enumerate(algorithms):\n            try:\n                method_name = getattr(algorithm, 'get_method_name', lambda: f'Method_{i+1}')()\n                result = self.evaluate_algorithm(algorithm, X, Y, method_name)\n                results[method_name] = result\n            except Exception as e:\n                print(f\"   ❌ 算法 {i+1} 评估失败: {e}\")\n                continue\n        \n        # 生成比较报告\n        comparison_report = self._generate_comparison_report(results)\n        \n        return {\n            'individual_results': results,\n            'comparison_report': comparison_report\n        }\n    \n    def _generate_comparison_report(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:\n        \"\"\"\n        生成比较报告\n        \n        Args:\n            results: 算法结果字典\n            \n        Returns:\n            比较报告\n        \"\"\"\n        successful_results = {k: v for k, v in results.items() if v['success']}\n        \n        if not successful_results:\n            return {'error': '没有成功的算法结果'}\n        \n        # 找出最佳方法\n        best_method = max(successful_results.keys(), \n                         key=lambda k: successful_results[k]['best_accuracy'])\n        best_accuracy = successful_results[best_method]['best_accuracy']\n        \n        # 计算相对改进\n        baseline_method = list(successful_results.keys())[0]  # 使用第一个作为基准\n        baseline_accuracy = successful_results[baseline_method]['best_accuracy']\n        \n        improvements = {}\n        for method, result in successful_results.items():\n            if method != baseline_method:\n                if baseline_accuracy > 0:\n                    improvement = (result['best_accuracy'] - baseline_accuracy) / baseline_accuracy * 100\n                else:\n                    improvement = 0\n                improvements[method] = improvement\n        \n        # 效率分析\n        efficiency_analysis = {}\n        baseline_time = successful_results[baseline_method]['training_time']\n        for method, result in successful_results.items():\n            if result['training_time'] > 0:\n                speedup = baseline_time / result['training_time']\n            else:\n                speedup = 0\n            efficiency_analysis[method] = {\n                'training_time': result['training_time'],\n                'speedup': speedup\n            }\n        \n        return {\n            'total_methods': len(results),\n            'successful_methods': len(successful_results),\n            'best_method': best_method,\n            'best_accuracy': best_accuracy,\n            'baseline_method': baseline_method,\n            'baseline_accuracy': baseline_accuracy,\n            'improvements': improvements,\n            'efficiency_analysis': efficiency_analysis\n        }\n    \n    def print_comparison_report(self, comparison_data: Dict[str, Any]) -> None:\n        \"\"\"\n        打印比较报告\n        \n        Args:\n            comparison_data: 比较数据\n        \"\"\"\n        report = comparison_data['comparison_report']\n        results = comparison_data['individual_results']\n        \n        if 'error' in report:\n            print(f\"❌ {report['error']}\")\n            return\n        \n        print(\"\\n\" + \"=\" * 80)\n        print(\"📊 算法比较报告\")\n        print(\"=\" * 80)\n        \n        # 结果汇总表\n        print(f\"\\n{'方法':<25} {'成功':<6} {'准确率':<10} {'F1-Micro':<10} {'F1-Macro':<10} {'时间(s)':<10}\")\n        print(\"-\" * 80)\n        \n        for method, result in results.items():\n            status = \"✅\" if result['success'] else \"❌\"\n            accuracy = result.get('best_accuracy', 0)\n            f1_micro = result.get('best_f1_micro', 0)\n            f1_macro = result.get('best_f1_macro', 0)\n            time_taken = result.get('training_time', 0)\n            \n            print(f\"{method:<25} {status:<6} {accuracy:<10.4f} {f1_micro:<10.4f} {f1_macro:<10.4f} {time_taken:<10.2f}\")\n        \n        # 最佳方法\n        print(f\"\\n🏆 最佳方法: {report['best_method']}\")\n        print(f\"🏆 最佳准确率: {report['best_accuracy']:.4f}\")\n        \n        # 相对改进\n        if report['improvements']:\n            print(f\"\\n📈 相对 {report['baseline_method']} 的改进:\")\n            for method, improvement in report['improvements'].items():\n                emoji = \"📈\" if improvement > 0 else \"📉\"\n                print(f\"   {emoji} {method}: {improvement:+.1f}%\")\n        \n        # 效率分析\n        print(f\"\\n⚡ 效率分析:\")\n        for method, efficiency in report['efficiency_analysis'].items():\n            print(f\"   {method}: {efficiency['training_time']:.2f}s ({efficiency['speedup']:.1f}x)\")\n\nif __name__ == \"__main__\":\n    # 测试评估器\n    evaluator = Evaluator()\n    print(\"评估器初始化成功\")\n    print(f\"支持的分类器: {list(evaluator.classifier_map.keys())}\")
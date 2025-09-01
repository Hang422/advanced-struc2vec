#!/usr/bin/env python3
"""
USA Airports Dataset Experiment: Ensemble vs Original Struc2Vec
Specific experiment comparing advanced fusion methods with original Struc2Vec on USA airports dataset.

Dataset Info:
- Nodes: 1190 airports
- Edges: 13599 connections
- Classes: 4 regional groups (297, 297, 297, 299 airports each)
- Task: Node classification for regional airport grouping
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import time
import os
import sys

# Add project root to path to import your methods
sys.path.append('/Users/luohang/PycharmProjects/graphlet_struc2vec/advancedStruc2vec')

class USAAirportsExperiment:
    """
    Comprehensive experiment comparing Ensemble and Original Struc2Vec on USA airports.
    """
    
    def __init__(self, data_dir="data/raw/data/flight"):
        self.data_dir = data_dir
        self.results = {}
        self.graph = None
        self.labels = None
        self.node_mapping = None
        
    def load_usa_airports_data(self):
        """Load and preprocess USA airports dataset."""
        print("Loading USA Airports dataset...")
        
        # Load graph
        edge_file = f"{self.data_dir}/usa-airports.edgelist"
        label_file = f"{self.data_dir}/labels-usa-airports.txt"
        
        # Read edges
        edges = []
        with open(edge_file, 'r') as f:
            for line in f:
                u, v = map(int, line.strip().split())
                edges.append((u, v))
        
        # Read labels
        labels_dict = {}
        with open(label_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                node, label = line.strip().split()
                labels_dict[int(node)] = int(label)
        
        # Create graph
        self.graph = nx.Graph()
        self.graph.add_edges_from(edges)
        
        # Ensure we only keep nodes that have labels
        labeled_nodes = set(labels_dict.keys())
        nodes_in_graph = set(self.graph.nodes())
        valid_nodes = labeled_nodes.intersection(nodes_in_graph)
        
        # Filter graph to only include labeled nodes
        self.graph = self.graph.subgraph(valid_nodes).copy()
        
        # Create node mapping and labels array
        self.node_mapping = {node: i for i, node in enumerate(sorted(valid_nodes))}
        self.labels = np.array([labels_dict[node] for node in sorted(valid_nodes)])
        
        # Relabel graph nodes to be sequential
        self.graph = nx.relabel_nodes(self.graph, self.node_mapping)
        
        print(f"Dataset loaded:")
        print(f"  Nodes: {self.graph.number_of_nodes()}")
        print(f"  Edges: {self.graph.number_of_edges()}")
        print(f"  Classes: {len(np.unique(self.labels))}")
        print(f"  Class distribution: {np.bincount(self.labels)}")
        
        return self.graph, self.labels
    
    def dummy_original_struc2vec(self, graph, embedding_dim=128):
        """
        Placeholder for original Struc2Vec method.
        Replace this with actual implementation.
        """
        print("Running Original Struc2Vec (placeholder)...")
        
        # Simulate computation time
        time.sleep(0.5)
        
        # Generate embeddings based on graph structure (degree-based simulation)
        n_nodes = len(graph.nodes())
        embeddings = np.random.randn(n_nodes, embedding_dim)
        
        # Add some structure based on node degrees
        degrees = np.array([graph.degree(node) for node in graph.nodes()])
        degrees_normalized = (degrees - degrees.mean()) / (degrees.std() + 1e-8)
        
        # Inject degree information into first few dimensions
        embeddings[:, 0] = degrees_normalized
        embeddings[:, 1] = degrees_normalized ** 2
        
        return embeddings
    
    def dummy_ensemble_struc2vec(self, graph, embedding_dim=128):
        """
        Placeholder for ensemble/advanced Struc2Vec method.
        Replace this with actual implementation.
        """
        print("Running Ensemble Struc2Vec (placeholder)...")
        
        # Simulate longer computation time (more sophisticated method)
        time.sleep(1.0)
        
        n_nodes = len(graph.nodes())
        
        # Generate base embeddings
        embeddings = np.random.randn(n_nodes, embedding_dim)
        
        # Add structural features (simulating graphlet features, etc.)
        degrees = np.array([graph.degree(node) for node in graph.nodes()])
        clustering = np.array([nx.clustering(graph, node) for node in graph.nodes()])
        
        # Normalize features
        degrees_norm = (degrees - degrees.mean()) / (degrees.std() + 1e-8)
        clustering_norm = (clustering - clustering.mean()) / (clustering.std() + 1e-8)
        
        # Inject more sophisticated structural information
        embeddings[:, 0] = degrees_norm
        embeddings[:, 1] = clustering_norm
        embeddings[:, 2] = degrees_norm * clustering_norm  # Interaction term
        embeddings[:, 3] = np.array([graph.degree(node)**0.5 for node in graph.nodes()])
        
        # Add some ensemble-like random variations
        for i in range(5):
            noise_component = np.random.randn(n_nodes, embedding_dim // 8)
            start_idx = 4 + i * (embedding_dim // 8)
            end_idx = min(start_idx + embedding_dim // 8, embedding_dim)
            if start_idx < embedding_dim:
                embeddings[:, start_idx:end_idx] = noise_component[:, :end_idx-start_idx]
        
        return embeddings
    
    def evaluate_embeddings(self, embeddings, labels, method_name, test_size=0.3):
        """Comprehensive evaluation of embeddings."""
        print(f"\nEvaluating {method_name}...")
        
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings_scaled, labels, test_size=test_size, 
            random_state=42, stratify=labels
        )
        
        # Test multiple classifiers
        classifiers = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        results = {}
        
        for clf_name, clf in classifiers.items():
            print(f"  Testing with {clf_name}...")
            
            # Train and predict
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test) if hasattr(clf, 'predict_proba') else None
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_macro': f1_score(y_test, y_pred, average='macro'),
                'f1_micro': f1_score(y_test, y_pred, average='micro'),
                'precision_macro': precision_score(y_test, y_pred, average='macro'),
                'recall_macro': recall_score(y_test, y_pred, average='macro')
            }
            
            # Cross-validation
            cv_scores = cross_val_score(clf, embeddings_scaled, labels, 
                                      cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                                      scoring='accuracy')
            metrics['cv_accuracy_mean'] = cv_scores.mean()
            metrics['cv_accuracy_std'] = cv_scores.std()
            
            results[clf_name] = metrics
            
            # Print results
            print(f"    Accuracy: {metrics['accuracy']:.4f}")
            print(f"    F1-macro: {metrics['f1_macro']:.4f}")
            print(f"    CV Accuracy: {metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
        
        return results
    
    def visualize_embeddings(self, embeddings_dict, labels, save_dir="usa_airports_results"):
        """Create comprehensive visualizations."""
        os.makedirs(save_dir, exist_ok=True)
        
        print("\nCreating visualizations...")
        
        # 1. t-SNE visualization comparison
        fig, axes = plt.subplots(1, len(embeddings_dict), figsize=(15, 5))
        if len(embeddings_dict) == 1:
            axes = [axes]
        
        for i, (method_name, embeddings) in enumerate(embeddings_dict.items()):
            print(f"  Computing t-SNE for {method_name}...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            embeddings_2d = tsne.fit_transform(embeddings)
            
            # Plot
            scatter = axes[i].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                    c=labels, cmap='tab10', alpha=0.7, s=20)
            axes[i].set_title(f'{method_name}\nt-SNE Visualization')
            axes[i].grid(True, alpha=0.3)
            
            if i == len(embeddings_dict) - 1:  # Add colorbar to last plot
                cbar = plt.colorbar(scatter, ax=axes[i])
                cbar.set_label('Airport Region')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/embeddings_tsne_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. PCA comparison
        fig, axes = plt.subplots(1, len(embeddings_dict), figsize=(15, 5))
        if len(embeddings_dict) == 1:
            axes = [axes]
        
        for i, (method_name, embeddings) in enumerate(embeddings_dict.items()):
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)
            
            scatter = axes[i].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                    c=labels, cmap='tab10', alpha=0.7, s=20)
            axes[i].set_title(f'{method_name}\nPCA Visualization')
            axes[i].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
            axes[i].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
            axes[i].grid(True, alpha=0.3)
            
            if i == len(embeddings_dict) - 1:
                cbar = plt.colorbar(scatter, ax=axes[i])
                cbar.set_label('Airport Region')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/embeddings_pca_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_performance_comparison(self, results_dict, save_dir="usa_airports_results"):
        """Create performance comparison visualizations."""
        os.makedirs(save_dir, exist_ok=True)
        
        print("\nCreating performance comparison charts...")
        
        # Extract data for plotting
        methods = list(results_dict.keys())
        classifiers = list(next(iter(results_dict.values())).keys())
        metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            data = []
            method_labels = []
            classifier_labels = []
            
            for method in methods:
                for classifier in classifiers:
                    data.append(results_dict[method][classifier][metric])
                    method_labels.append(method)
                    classifier_labels.append(classifier)
            
            # Create grouped bar plot
            x_pos = np.arange(len(classifiers))
            width = 0.35
            
            method1_data = [results_dict[methods[0]][clf][metric] for clf in classifiers]
            method2_data = [results_dict[methods[1]][clf][metric] for clf in classifiers] if len(methods) > 1 else []
            
            bars1 = axes[i].bar(x_pos - width/2, method1_data, width, 
                              label=methods[0], alpha=0.8)
            if method2_data:
                bars2 = axes[i].bar(x_pos + width/2, method2_data, width, 
                                  label=methods[1], alpha=0.8)
            
            axes[i].set_xlabel('Classifier')
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
            axes[i].set_xticks(x_pos)
            axes[i].set_xticklabels(classifiers, rotation=15)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            def autolabel(bars):
                for bar in bars:
                    height = bar.get_height()
                    axes[i].annotate(f'{height:.3f}',
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 3),
                                   textcoords="offset points",
                                   ha='center', va='bottom', fontsize=9)
            
            autolabel(bars1)
            if method2_data:
                autolabel(bars2)
        
        plt.suptitle('USA Airports: Method Comparison Across Classifiers', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_experiment(self):
        """Run the complete experimental pipeline."""
        print("="*60)
        print("USA AIRPORTS EXPERIMENT: Ensemble vs Original Struc2Vec")
        print("="*60)
        
        # 1. Load data
        graph, labels = self.load_usa_airports_data()
        
        # 2. Run methods and measure time
        print("\n" + "="*40)
        print("RUNNING EMBEDDING METHODS")
        print("="*40)
        
        embeddings_dict = {}
        timing_results = {}
        
        # Original Struc2Vec
        start_time = time.time()
        original_embeddings = self.dummy_original_struc2vec(graph)
        original_time = time.time() - start_time
        embeddings_dict['Original Struc2Vec'] = original_embeddings
        timing_results['Original Struc2Vec'] = original_time
        
        # Ensemble Struc2Vec
        start_time = time.time()
        ensemble_embeddings = self.dummy_ensemble_struc2vec(graph)
        ensemble_time = time.time() - start_time
        embeddings_dict['Ensemble Struc2Vec'] = ensemble_embeddings
        timing_results['Ensemble Struc2Vec'] = ensemble_time
        
        print(f"\nTiming Results:")
        for method, time_taken in timing_results.items():
            print(f"  {method}: {time_taken:.2f} seconds")
        
        # 3. Evaluate methods
        print("\n" + "="*40)
        print("EVALUATION RESULTS")
        print("="*40)
        
        all_results = {}
        for method_name, embeddings in embeddings_dict.items():
            results = self.evaluate_embeddings(embeddings, labels, method_name)
            all_results[method_name] = results
        
        # 4. Create visualizations
        print("\n" + "="*40)
        print("CREATING VISUALIZATIONS")
        print("="*40)
        
        self.visualize_embeddings(embeddings_dict, labels)
        self.create_performance_comparison(all_results)
        
        # 5. Summary report
        self.create_summary_report(all_results, timing_results)
        
        # Store results
        self.results = {
            'embeddings': embeddings_dict,
            'performance': all_results,
            'timing': timing_results,
            'dataset_info': {
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
                'classes': len(np.unique(labels)),
                'class_distribution': np.bincount(labels).tolist()
            }
        }
        
        return self.results
    
    def create_summary_report(self, results_dict, timing_dict, save_dir="usa_airports_results"):
        """Create a comprehensive summary report."""
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("EXPERIMENTAL SUMMARY REPORT")
        print("="*60)
        
        report = []
        report.append("USA Airports Dataset Experiment Summary")
        report.append("="*50)
        report.append("")
        
        # Dataset info
        report.append("Dataset Information:")
        report.append(f"  Nodes: {self.graph.number_of_nodes()}")
        report.append(f"  Edges: {self.graph.number_of_edges()}")
        report.append(f"  Classes: {len(np.unique(self.labels))}")
        report.append(f"  Class distribution: {np.bincount(self.labels)}")
        report.append("")
        
        # Timing comparison
        report.append("Computation Time Comparison:")
        for method, time_taken in timing_dict.items():
            report.append(f"  {method}: {time_taken:.2f} seconds")
        report.append("")
        
        # Performance comparison
        report.append("Performance Comparison (Best scores for each classifier):")
        classifiers = list(next(iter(results_dict.values())).keys())
        
        for classifier in classifiers:
            report.append(f"\n{classifier}:")
            report.append("-" * (len(classifier) + 1))
            
            metrics = ['accuracy', 'f1_macro', 'cv_accuracy_mean']
            for metric in metrics:
                scores = {}
                for method in results_dict.keys():
                    scores[method] = results_dict[method][classifier][metric]
                
                best_method = max(scores, key=scores.get)
                report.append(f"  {metric}: {best_method} ({scores[best_method]:.4f})")
                
                # Show differences
                method_scores = [(method, score) for method, score in scores.items()]
                method_scores.sort(key=lambda x: x[1], reverse=True)
                if len(method_scores) > 1:
                    diff = method_scores[0][1] - method_scores[1][1]
                    report.append(f"    Improvement: +{diff:.4f} ({diff/method_scores[1][1]*100:.1f}%)")
        
        # Overall conclusion
        report.append("\nOverall Analysis:")
        
        # Calculate average performance across all classifiers
        method_avg_scores = {}
        for method in results_dict.keys():
            total_score = 0
            count = 0
            for classifier in classifiers:
                total_score += results_dict[method][classifier]['accuracy']
                count += 1
            method_avg_scores[method] = total_score / count
        
        best_overall = max(method_avg_scores, key=method_avg_scores.get)
        report.append(f"  Best overall method: {best_overall}")
        report.append(f"  Average accuracy: {method_avg_scores[best_overall]:.4f}")
        
        if len(method_avg_scores) > 1:
            methods = list(method_avg_scores.keys())
            other_method = methods[1] if methods[0] == best_overall else methods[0]
            improvement = method_avg_scores[best_overall] - method_avg_scores[other_method]
            report.append(f"  Improvement over {other_method}: +{improvement:.4f}")
        
        # Print and save report
        report_text = "\n".join(report)
        print(report_text)
        
        with open(f"{save_dir}/experiment_summary.txt", 'w') as f:
            f.write(report_text)
        
        print(f"\nReport saved to {save_dir}/experiment_summary.txt")


def main():
    """Run the USA Airports experiment."""
    experiment = USAAirportsExperiment()
    results = experiment.run_complete_experiment()
    
    print("\n🎉 Experiment completed successfully!")
    print("Check the 'usa_airports_results' directory for visualizations and reports.")
    
    return results


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Quick Test Version: USA Airports Experiment
Fast version for testing the experimental framework.
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import time
import os

class QuickUSAExperiment:
    """Quick test version of USA airports experiment."""
    
    def __init__(self):
        self.graph = None
        self.labels = None
    
    def load_data(self):
        """Load USA airports data."""
        print("Loading USA Airports dataset...")
        
        # Load graph
        edges = []
        with open("data/raw/data/flight/usa-airports.edgelist", 'r') as f:
            for line in f:
                u, v = map(int, line.strip().split())
                edges.append((u, v))
        
        # Load labels
        labels_dict = {}
        with open("data/raw/data/flight/labels-usa-airports.txt", 'r') as f:
            next(f)  # Skip header
            for line in f:
                node, label = line.strip().split()
                labels_dict[int(node)] = int(label)
        
        # Create graph
        self.graph = nx.Graph()
        self.graph.add_edges_from(edges)
        
        # Filter to labeled nodes only
        labeled_nodes = set(labels_dict.keys())
        nodes_in_graph = set(self.graph.nodes())
        valid_nodes = labeled_nodes.intersection(nodes_in_graph)
        
        self.graph = self.graph.subgraph(valid_nodes).copy()
        
        # Create sequential node mapping
        node_mapping = {node: i for i, node in enumerate(sorted(valid_nodes))}
        self.graph = nx.relabel_nodes(self.graph, node_mapping)
        self.labels = np.array([labels_dict[node] for node in sorted(valid_nodes)])
        
        print(f"Dataset loaded: {len(self.graph)} nodes, {self.graph.number_of_edges()} edges")
        print(f"Class distribution: {np.bincount(self.labels)}")
        
        return self.graph, self.labels
    
    def create_embeddings(self):
        """Create dummy embeddings for both methods."""
        print("\nGenerating embeddings...")
        n_nodes = len(self.graph)
        embedding_dim = 64  # Reduced dimension for speed
        
        # Calculate structural features
        degrees = np.array([self.graph.degree(node) for node in self.graph.nodes()])
        clustering = np.array([nx.clustering(self.graph, node) for node in self.graph.nodes()])
        
        # Normalize
        degrees_norm = (degrees - degrees.mean()) / (degrees.std() + 1e-8)
        clustering_norm = (clustering - clustering.mean()) / (clustering.std() + 1e-8)
        
        # Original Struc2Vec (simulated)
        print("  Creating Original Struc2Vec embeddings...")
        original_emb = np.random.randn(n_nodes, embedding_dim)
        original_emb[:, 0] = degrees_norm
        original_emb[:, 1] = degrees_norm ** 2
        
        # Ensemble Struc2Vec (simulated with more features)
        print("  Creating Ensemble Struc2Vec embeddings...")
        ensemble_emb = np.random.randn(n_nodes, embedding_dim)
        ensemble_emb[:, 0] = degrees_norm
        ensemble_emb[:, 1] = clustering_norm
        ensemble_emb[:, 2] = degrees_norm * clustering_norm
        ensemble_emb[:, 3] = np.sqrt(degrees + 1)
        
        # Add betweenness centrality for a subset of nodes (for speed)
        if len(self.graph) <= 500:
            betweenness = nx.betweenness_centrality(self.graph)
            betweenness_values = np.array([betweenness[node] for node in self.graph.nodes()])
            betweenness_norm = (betweenness_values - betweenness_values.mean()) / (betweenness_values.std() + 1e-8)
            ensemble_emb[:, 4] = betweenness_norm
        
        return {
            'Original Struc2Vec': original_emb,
            'Ensemble Struc2Vec': ensemble_emb
        }
    
    def evaluate_methods(self, embeddings_dict):
        """Quick evaluation of both methods."""
        print("\nEvaluating methods...")
        
        results = {}
        
        for method_name, embeddings in embeddings_dict.items():
            print(f"  Evaluating {method_name}...")
            
            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(embeddings)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, self.labels, test_size=0.3, random_state=42, stratify=self.labels
            )
            
            # Simple logistic regression
            clf = LogisticRegression(random_state=42, max_iter=500)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            
            results[method_name] = {
                'accuracy': accuracy,
                'f1_macro': f1,
                'test_size': len(y_test)
            }
            
            print(f"    Accuracy: {accuracy:.4f}")
            print(f"    F1-macro: {f1:.4f}")
        
        return results
    
    def create_visualizations(self, embeddings_dict):
        """Create quick visualizations."""
        print("\nCreating visualizations...")
        
        os.makedirs("quick_results", exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # t-SNE plots
        for i, (method_name, embeddings) in enumerate(embeddings_dict.items()):
            print(f"  Computing t-SNE for {method_name}...")
            
            # Use smaller perplexity and fewer iterations for speed
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//4), n_iter=300)
            embeddings_2d = tsne.fit_transform(embeddings[:500])  # Subsample for speed
            labels_sub = self.labels[:500]
            
            scatter = axes[0, i].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                       c=labels_sub, cmap='tab10', alpha=0.7, s=15)
            axes[0, i].set_title(f'{method_name}\nt-SNE Visualization')
            axes[0, i].grid(True, alpha=0.3)
        
        # PCA plots
        for i, (method_name, embeddings) in enumerate(embeddings_dict.items()):
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings[:500])
            labels_sub = self.labels[:500]
            
            scatter = axes[1, i].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                       c=labels_sub, cmap='tab10', alpha=0.7, s=15)
            axes[1, i].set_title(f'{method_name}\nPCA Visualization')
            axes[1, i].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
            axes[1, i].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
            axes[1, i].grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(scatter, ax=axes, label='Airport Region')
        
        plt.tight_layout()
        plt.savefig("quick_results/embeddings_comparison.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    def create_performance_chart(self, results):
        """Create performance comparison chart."""
        print("Creating performance comparison...")
        
        methods = list(results.keys())
        accuracies = [results[method]['accuracy'] for method in methods]
        f1_scores = [results[method]['f1_macro'] for method in methods]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy comparison
        bars1 = ax1.bar(methods, accuracies, alpha=0.8, color=['skyblue', 'lightcoral'])
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Classification Accuracy Comparison')
        ax1.set_ylim(0, max(accuracies) * 1.1)
        
        # Add value labels
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # F1 comparison
        bars2 = ax2.bar(methods, f1_scores, alpha=0.8, color=['skyblue', 'lightcoral'])
        ax2.set_ylabel('F1-Macro Score')
        ax2.set_title('F1-Macro Score Comparison')
        ax2.set_ylim(0, max(f1_scores) * 1.1)
        
        for bar, f1 in zip(bars2, f1_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{f1:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig("quick_results/performance_comparison.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    def run_experiment(self):
        """Run the complete quick experiment."""
        print("=" * 50)
        print("QUICK USA AIRPORTS EXPERIMENT")
        print("=" * 50)
        
        # Load data
        self.load_data()
        
        # Create embeddings
        embeddings_dict = self.create_embeddings()
        
        # Evaluate
        results = self.evaluate_methods(embeddings_dict)
        
        # Visualize
        self.create_visualizations(embeddings_dict)
        self.create_performance_chart(results)
        
        # Summary
        print("\n" + "=" * 30)
        print("EXPERIMENT SUMMARY")
        print("=" * 30)
        
        print(f"Dataset: {len(self.graph)} airports, {self.graph.number_of_edges()} routes")
        print(f"Classes: {len(np.unique(self.labels))} regions")
        
        print("\nPerformance Results:")
        for method, metrics in results.items():
            print(f"  {method}:")
            print(f"    Accuracy: {metrics['accuracy']:.4f}")
            print(f"    F1-macro: {metrics['f1_macro']:.4f}")
        
        # Determine winner
        best_method = max(results.keys(), key=lambda x: results[x]['accuracy'])
        improvement = results[best_method]['accuracy'] - min(results[method]['accuracy'] for method in results.keys())
        
        print(f"\nBest Method: {best_method}")
        print(f"Improvement: +{improvement:.4f} accuracy")
        
        print("\n✅ Quick experiment completed!")
        print("Results saved in 'quick_results' directory")
        
        return results


if __name__ == "__main__":
    experiment = QuickUSAExperiment()
    results = experiment.run_experiment()
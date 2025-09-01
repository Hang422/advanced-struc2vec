#!/usr/bin/env python3
"""
Advanced Visualization Experiments for Struc2Vec Analysis
Based on insights from the original Struc2Vec paper for comprehensive visual analysis.

This module provides various visualization tools to analyze:
1. Structural similarity hierarchies
2. Embedding quality assessment
3. Multilayer graph structure
4. Performance comparisons
5. Network structural analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
import umap
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class Struc2VecVisualizer:
    """
    Comprehensive visualization toolkit for Struc2Vec and Advanced Struc2Vec analysis.
    """
    
    def __init__(self, output_dir="visualizations"):
        self.output_dir = output_dir
        self.setup_plotting()
    
    def setup_plotting(self):
        """Setup matplotlib and seaborn styles."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
    
    def visualize_embedding_space(self, embeddings, labels=None, method='tsne', 
                                title="Embedding Space Visualization", save=True):
        """
        Visualize embeddings in 2D space using various dimensionality reduction techniques.
        
        Args:
            embeddings: Node embeddings matrix (n_nodes x embedding_dim)
            labels: Node labels for coloring
            method: 'tsne', 'umap', 'pca', 'mds'
            title: Plot title
            save: Whether to save the plot
        """
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        elif method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        elif method == 'mds':
            reducer = MDS(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        embedding_2d = reducer.fit_transform(embeddings)
        
        plt.figure(figsize=(12, 8))
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            for i, label in enumerate(unique_labels):
                mask = labels == label
                plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1], 
                          c=[colors[i]], label=f'Class {label}', alpha=0.7, s=60)
            plt.legend()
        else:
            plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.7, s=60)
        
        plt.title(f"{title} ({method.upper()})")
        plt.xlabel(f"{method.upper()} Component 1")
        plt.ylabel(f"{method.upper()} Component 2")
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(f"{self.output_dir}/embedding_{method}_{title.lower().replace(' ', '_')}.png", 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def structural_similarity_heatmap(self, similarity_matrix, node_labels=None, 
                                   title="Structural Similarity Matrix"):
        """
        Create heatmap of structural similarity between nodes.
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            node_labels: Labels for nodes
            title: Plot title
        """
        plt.figure(figsize=(12, 10))
        
        if node_labels is not None:
            # Sort by labels for better visualization
            sort_idx = np.argsort(node_labels)
            similarity_matrix = similarity_matrix[sort_idx][:, sort_idx]
        
        sns.heatmap(similarity_matrix, cmap='viridis', square=True, 
                   cbar_kws={'label': 'Structural Similarity'})
        
        plt.title(title)
        plt.xlabel("Node Index")
        plt.ylabel("Node Index")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/similarity_heatmap.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def hierarchical_similarity_analysis(self, multilayer_similarities, k_max=5):
        """
        Visualize how structural similarity changes across different layers (k-hop neighborhoods).
        
        Args:
            multilayer_similarities: Dict with k as keys and similarity matrices as values
            k_max: Maximum k to visualize
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, k in enumerate(range(min(k_max, len(multilayer_similarities)))):
            if k in multilayer_similarities:
                sim_matrix = multilayer_similarities[k]
                im = axes[i].imshow(sim_matrix, cmap='viridis', aspect='auto')
                axes[i].set_title(f"Layer {k} (k-hop neighborhood)")
                axes[i].set_xlabel("Node Index")
                axes[i].set_ylabel("Node Index")
                plt.colorbar(im, ax=axes[i])
        
        # Hide unused subplots
        for i in range(len(multilayer_similarities), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle("Structural Similarity Across Different Layers", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/hierarchical_similarity.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def embedding_quality_metrics(self, embeddings, true_similarities, labels=None):
        """
        Analyze embedding quality through various metrics and visualizations.
        
        Args:
            embeddings: Node embeddings
            true_similarities: Ground truth similarity matrix
            labels: Node labels for analysis
        """
        # Calculate embedding distances
        embedding_distances = squareform(pdist(embeddings, metric='euclidean'))
        
        # Flatten matrices for correlation analysis
        true_sim_flat = true_similarities[np.triu_indices_from(true_similarities, k=1)]
        emb_dist_flat = embedding_distances[np.triu_indices_from(embedding_distances, k=1)]
        
        # Calculate correlations
        pearson_corr, _ = pearsonr(true_sim_flat, -emb_dist_flat)  # Negative because similarity vs distance
        spearman_corr, _ = spearmanr(true_sim_flat, -emb_dist_flat)
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Scatter plot: True similarity vs Embedding distance
        ax1.scatter(true_sim_flat, emb_dist_flat, alpha=0.5)
        ax1.set_xlabel("True Structural Similarity")
        ax1.set_ylabel("Embedding Distance")
        ax1.set_title(f"Similarity vs Distance\nPearson: {pearson_corr:.3f}, Spearman: {spearman_corr:.3f}")
        ax1.grid(True, alpha=0.3)
        
        # Distribution of similarities
        ax2.hist(true_sim_flat, bins=50, alpha=0.7, label='True Similarities')
        ax2.hist(-emb_dist_flat, bins=50, alpha=0.7, label='Embedding Similarities')
        ax2.set_xlabel("Similarity Value")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Similarity Distributions")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Embedding distance distribution
        ax3.hist(emb_dist_flat, bins=50, alpha=0.7, color='orange')
        ax3.set_xlabel("Embedding Distance")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Embedding Distance Distribution")
        ax3.grid(True, alpha=0.3)
        
        # If labels are provided, analyze within-class vs between-class distances
        if labels is not None:
            within_class_distances = []
            between_class_distances = []
            
            for i in range(len(labels)):
                for j in range(i+1, len(labels)):
                    if labels[i] == labels[j]:
                        within_class_distances.append(embedding_distances[i, j])
                    else:
                        between_class_distances.append(embedding_distances[i, j])
            
            ax4.hist(within_class_distances, bins=30, alpha=0.7, label='Within-class', density=True)
            ax4.hist(between_class_distances, bins=30, alpha=0.7, label='Between-class', density=True)
            ax4.set_xlabel("Embedding Distance")
            ax4.set_ylabel("Density")
            ax4.set_title("Within vs Between Class Distances")
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/embedding_quality_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'pearson_correlation': pearson_corr,
            'spearman_correlation': spearman_corr,
            'mean_embedding_distance': np.mean(emb_dist_flat),
            'std_embedding_distance': np.std(emb_dist_flat)
        }
    
    def method_comparison_visualization(self, results_dict, metrics=['accuracy', 'f1', 'precision', 'recall']):
        """
        Visualize comparison between different methods (Original Struc2Vec, Advanced Struc2Vec, etc.)
        
        Args:
            results_dict: Dictionary with method names as keys and metric dictionaries as values
            metrics: List of metrics to compare
        """
        methods = list(results_dict.keys())
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                values = [results_dict[method].get(metric, 0) for method in methods]
                bars = axes[i].bar(methods, values, alpha=0.8)
                axes[i].set_title(f"{metric.capitalize()} Comparison")
                axes[i].set_ylabel(metric.capitalize())
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/method_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def network_structural_analysis(self, G, pos=None, title="Network Structure Analysis"):
        """
        Analyze and visualize network structural properties.
        
        Args:
            G: NetworkX graph
            pos: Node positions for visualization
            title: Plot title
        """
        if pos is None:
            pos = nx.spring_layout(G, k=1, iterations=50)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Network visualization
        degrees = dict(G.degree())
        node_sizes = [degrees[node] * 50 for node in G.nodes()]
        nx.draw(G, pos, ax=ax1, node_size=node_sizes, node_color='lightblue',
                edge_color='gray', alpha=0.7, with_labels=False)
        ax1.set_title("Network Structure (Node size ∝ degree)")
        
        # Degree distribution
        degree_values = list(degrees.values())
        ax2.hist(degree_values, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel("Degree")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Degree Distribution")
        ax2.grid(True, alpha=0.3)
        
        # Clustering coefficient distribution
        clustering = nx.clustering(G)
        clustering_values = list(clustering.values())
        ax3.hist(clustering_values, bins=20, alpha=0.7, edgecolor='black', color='orange')
        ax3.set_xlabel("Clustering Coefficient")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Clustering Coefficient Distribution")
        ax3.grid(True, alpha=0.3)
        
        # Betweenness centrality (for smaller graphs)
        if len(G.nodes()) <= 500:  # Only for smaller graphs due to computational cost
            betweenness = nx.betweenness_centrality(G)
            betweenness_values = list(betweenness.values())
            ax4.hist(betweenness_values, bins=20, alpha=0.7, edgecolor='black', color='green')
            ax4.set_xlabel("Betweenness Centrality")
            ax4.set_ylabel("Frequency")
            ax4.set_title("Betweenness Centrality Distribution")
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "Betweenness Centrality\n(Too large for computation)", 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/network_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print basic statistics
        print(f"Network Statistics:")
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        print(f"  Average degree: {np.mean(degree_values):.2f}")
        print(f"  Average clustering: {np.mean(clustering_values):.3f}")
        if nx.is_connected(G):
            print(f"  Average shortest path: {nx.average_shortest_path_length(G):.2f}")
        else:
            print(f"  Number of connected components: {nx.number_connected_components(G)}")
    
    def interactive_embedding_plot(self, embeddings, labels=None, node_names=None, 
                                 method='tsne', title="Interactive Embedding Visualization"):
        """
        Create interactive embedding visualization using Plotly.
        
        Args:
            embeddings: Node embeddings
            labels: Node labels
            node_names: Names for nodes (for hover text)
            method: Dimensionality reduction method
            title: Plot title
        """
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        elif method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        embedding_2d = reducer.fit_transform(embeddings)
        
        hover_text = []
        if node_names is not None:
            for i, name in enumerate(node_names):
                hover_text.append(f"Node: {name}<br>Index: {i}")
        else:
            hover_text = [f"Node {i}" for i in range(len(embeddings))]
        
        if labels is not None:
            fig = px.scatter(x=embedding_2d[:, 0], y=embedding_2d[:, 1], 
                           color=labels, hover_name=hover_text,
                           title=f"{title} ({method.upper()})",
                           labels={'x': f'{method.upper()} 1', 'y': f'{method.upper()} 2'})
        else:
            fig = px.scatter(x=embedding_2d[:, 0], y=embedding_2d[:, 1], 
                           hover_name=hover_text,
                           title=f"{title} ({method.upper()})",
                           labels={'x': f'{method.upper()} 1', 'y': f'{method.upper()} 2'})
        
        fig.update_layout(width=800, height=600)
        fig.write_html(f"{self.output_dir}/interactive_{method}_embedding.html")
        fig.show()
    
    def convergence_analysis(self, loss_history, title="Training Convergence"):
        """
        Visualize training convergence.
        
        Args:
            loss_history: List of loss values during training
            title: Plot title
        """
        plt.figure(figsize=(12, 6))
        plt.plot(loss_history, linewidth=2)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.output_dir}/convergence_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()


def create_sample_visualizations():
    """
    Create sample visualizations with synthetic data for demonstration.
    """
    visualizer = Struc2VecVisualizer()
    
    # Generate sample data
    n_nodes = 100
    embedding_dim = 128
    n_classes = 4
    
    # Sample embeddings
    embeddings = np.random.randn(n_nodes, embedding_dim)
    labels = np.random.randint(0, n_classes, n_nodes)
    
    # Sample similarity matrix
    similarity_matrix = np.random.rand(n_nodes, n_nodes)
    similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(similarity_matrix, 1.0)
    
    print("Creating sample visualizations...")
    
    # Embedding space visualization
    visualizer.visualize_embedding_space(embeddings, labels, method='tsne')
    
    # Similarity heatmap (smaller subset for visibility)
    subset_idx = np.random.choice(n_nodes, 30, replace=False)
    small_similarity = similarity_matrix[subset_idx][:, subset_idx]
    small_labels = labels[subset_idx]
    visualizer.structural_similarity_heatmap(small_similarity, small_labels)
    
    # Quality metrics
    metrics = visualizer.embedding_quality_metrics(embeddings, similarity_matrix, labels)
    print(f"Quality metrics: {metrics}")
    
    print("Sample visualizations created!")


if __name__ == "__main__":
    import os
    os.makedirs("visualizations", exist_ok=True)
    create_sample_visualizations()
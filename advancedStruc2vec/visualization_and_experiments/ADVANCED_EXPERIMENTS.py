#!/usr/bin/env python3
"""
Advanced Experimental Framework for Struc2Vec Analysis
Based on the original Struc2Vec paper with extensions for comprehensive evaluation.

This module provides advanced experiments including:
1. Robustness analysis under various noise conditions
2. Scalability experiments across different network sizes
3. Parameter sensitivity analysis
4. Ablation studies
5. Cross-domain evaluation
6. Temporal network analysis
7. Multi-layer network experiments
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class AdvancedStruc2VecExperiments:
    """
    Advanced experimental framework for comprehensive Struc2Vec analysis.
    """
    
    def __init__(self, output_dir="experimental_results"):
        self.output_dir = output_dir
        self.results = {}
    
    def robustness_edge_removal_experiment(self, G, embeddings_func, sampling_rates=None, 
                                         n_trials=10, labels=None):
        """
        Test robustness to edge removal using parsimonious edge sampling model.
        Based on Section 4.3 of the original paper.
        
        Args:
            G: Original graph
            embeddings_func: Function that takes a graph and returns embeddings
            sampling_rates: List of edge sampling probabilities
            n_trials: Number of trials per sampling rate
            labels: Node labels for classification evaluation
        """
        if sampling_rates is None:
            sampling_rates = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        
        results = {
            'sampling_rates': sampling_rates,
            'structural_preservation': [],
            'classification_performance': [],
            'std_preservation': [],
            'std_classification': []
        }
        
        # Get original embeddings
        original_embeddings = embeddings_func(G)
        original_similarity = self._compute_pairwise_distances(original_embeddings)
        
        print("Running robustness to edge removal experiment...")
        
        for rate in tqdm(sampling_rates):
            preservation_scores = []
            classification_scores = []
            
            for trial in range(n_trials):
                # Create sampled graph
                G_sampled = self._sample_edges(G, rate)
                
                # Ensure graph is connected (at least largest component)
                if not nx.is_connected(G_sampled):
                    largest_cc = max(nx.connected_components(G_sampled), key=len)
                    G_sampled = G_sampled.subgraph(largest_cc).copy()
                
                if G_sampled.number_of_nodes() < 10:  # Skip if too small
                    continue
                
                # Get embeddings for sampled graph
                try:
                    sampled_embeddings = embeddings_func(G_sampled)
                    sampled_similarity = self._compute_pairwise_distances(sampled_embeddings)
                    
                    # Compute structural preservation (correlation with original)
                    preservation = self._compute_preservation_score(original_similarity, 
                                                                   sampled_similarity, 
                                                                   list(G_sampled.nodes()))
                    preservation_scores.append(preservation)
                    
                    # Classification performance if labels available
                    if labels is not None:
                        sampled_labels = [labels[i] for i in G_sampled.nodes() if i in labels]
                        if len(set(sampled_labels)) > 1:  # Multi-class
                            clf_score = self._evaluate_classification(sampled_embeddings, sampled_labels)
                            classification_scores.append(clf_score)
                
                except Exception as e:
                    print(f"Error in trial {trial} for rate {rate}: {e}")
                    continue
            
            if preservation_scores:
                results['structural_preservation'].append(np.mean(preservation_scores))
                results['std_preservation'].append(np.std(preservation_scores))
            else:
                results['structural_preservation'].append(0)
                results['std_preservation'].append(0)
            
            if classification_scores:
                results['classification_performance'].append(np.mean(classification_scores))
                results['std_classification'].append(np.std(classification_scores))
            else:
                results['classification_performance'].append(0)
                results['std_classification'].append(0)
        
        self.results['robustness_edge_removal'] = results
        self._plot_robustness_results(results)
        return results
    
    def scalability_experiment(self, graph_generator, embeddings_func, 
                             sizes=[100, 200, 500, 1000, 2000, 5000]):
        """
        Test scalability across different network sizes.
        
        Args:
            graph_generator: Function that takes size and returns a graph
            embeddings_func: Function that takes a graph and returns embeddings
            sizes: List of network sizes to test
        """
        results = {
            'sizes': sizes,
            'computation_times': [],
            'memory_usage': [],
            'embedding_quality': []
        }
        
        print("Running scalability experiment...")
        
        for size in tqdm(sizes):
            try:
                # Generate graph
                G = graph_generator(size)
                
                # Measure computation time
                start_time = time.time()
                embeddings = embeddings_func(G)
                computation_time = time.time() - start_time
                
                results['computation_times'].append(computation_time)
                
                # Estimate memory usage (rough approximation)
                memory_estimate = embeddings.nbytes + G.number_of_edges() * 8
                results['memory_usage'].append(memory_estimate / (1024**2))  # MB
                
                # Compute embedding quality metric
                quality = self._compute_embedding_quality(G, embeddings)
                results['embedding_quality'].append(quality)
                
            except Exception as e:
                print(f"Error for size {size}: {e}")
                results['computation_times'].append(np.nan)
                results['memory_usage'].append(np.nan)
                results['embedding_quality'].append(np.nan)
        
        self.results['scalability'] = results
        self._plot_scalability_results(results)
        return results
    
    def parameter_sensitivity_analysis(self, G, embeddings_func_factory, 
                                     parameter_ranges, labels=None):
        """
        Analyze sensitivity to different parameters.
        
        Args:
            G: Input graph
            embeddings_func_factory: Function that takes parameters and returns embedding function
            parameter_ranges: Dictionary of parameter names to list of values
            labels: Node labels for evaluation
        """
        results = {}
        
        print("Running parameter sensitivity analysis...")
        
        for param_name, param_values in parameter_ranges.items():
            print(f"Analyzing parameter: {param_name}")
            
            param_results = {
                'values': param_values,
                'classification_scores': [],
                'structural_quality': [],
                'computation_times': []
            }
            
            for value in tqdm(param_values):
                try:
                    # Create embedding function with specific parameter value
                    params = {param_name: value}
                    embeddings_func = embeddings_func_factory(**params)
                    
                    # Measure computation time
                    start_time = time.time()
                    embeddings = embeddings_func(G)
                    computation_time = time.time() - start_time
                    
                    param_results['computation_times'].append(computation_time)
                    
                    # Evaluate classification performance
                    if labels is not None:
                        clf_score = self._evaluate_classification(embeddings, labels)
                        param_results['classification_scores'].append(clf_score)
                    
                    # Evaluate structural quality
                    quality = self._compute_embedding_quality(G, embeddings)
                    param_results['structural_quality'].append(quality)
                    
                except Exception as e:
                    print(f"Error for {param_name}={value}: {e}")
                    param_results['classification_scores'].append(np.nan)
                    param_results['structural_quality'].append(np.nan)
                    param_results['computation_times'].append(np.nan)
            
            results[param_name] = param_results
        
        self.results['parameter_sensitivity'] = results
        self._plot_parameter_sensitivity(results)
        return results
    
    def cross_domain_evaluation(self, datasets, embeddings_func, evaluation_metrics=None):
        """
        Evaluate performance across different types of networks.
        
        Args:
            datasets: Dictionary of dataset_name -> (graph, labels)
            embeddings_func: Embedding function
            evaluation_metrics: List of metrics to compute
        """
        if evaluation_metrics is None:
            evaluation_metrics = ['accuracy', 'f1_macro', 'silhouette']
        
        results = {
            'datasets': list(datasets.keys()),
            'metrics': {}
        }
        
        for metric in evaluation_metrics:
            results['metrics'][metric] = []
        
        print("Running cross-domain evaluation...")
        
        for dataset_name, (G, labels) in datasets.items():
            print(f"Evaluating on {dataset_name}...")
            
            try:
                # Get embeddings
                embeddings = embeddings_func(G)
                
                # Compute metrics
                for metric in evaluation_metrics:
                    if metric == 'accuracy':
                        score = self._evaluate_classification(embeddings, labels, metric='accuracy')
                    elif metric == 'f1_macro':
                        score = self._evaluate_classification(embeddings, labels, metric='f1_macro')
                    elif metric == 'silhouette':
                        score = self._compute_silhouette_score(embeddings, labels)
                    else:
                        score = np.nan
                    
                    results['metrics'][metric].append(score)
                
            except Exception as e:
                print(f"Error evaluating {dataset_name}: {e}")
                for metric in evaluation_metrics:
                    results['metrics'][metric].append(np.nan)
        
        self.results['cross_domain'] = results
        self._plot_cross_domain_results(results)
        return results
    
    def ablation_study(self, G, base_embeddings_func, ablation_configs, labels=None):
        """
        Conduct ablation study by removing/modifying different components.
        
        Args:
            G: Input graph
            base_embeddings_func: Base embedding function
            ablation_configs: Dictionary of configuration_name -> embedding_function
            labels: Node labels for evaluation
        """
        results = {
            'configurations': list(ablation_configs.keys()),
            'performance': []
        }
        
        print("Running ablation study...")
        
        for config_name, embeddings_func in ablation_configs.items():
            print(f"Evaluating configuration: {config_name}")
            
            try:
                embeddings = embeddings_func(G)
                
                if labels is not None:
                    performance = self._evaluate_classification(embeddings, labels)
                else:
                    performance = self._compute_embedding_quality(G, embeddings)
                
                results['performance'].append(performance)
                
            except Exception as e:
                print(f"Error in configuration {config_name}: {e}")
                results['performance'].append(0)
        
        self.results['ablation'] = results
        self._plot_ablation_results(results)
        return results
    
    def structural_role_analysis(self, G, embeddings, ground_truth_roles=None):
        """
        Analyze how well embeddings capture different structural roles.
        
        Args:
            G: Input graph
            embeddings: Node embeddings
            ground_truth_roles: Known structural roles (if available)
        """
        results = {
            'node_roles': {},
            'role_separation': [],
            'role_consistency': []
        }
        
        # Compute structural features
        degrees = dict(G.degree())
        clustering = nx.clustering(G)
        
        if len(G) <= 1000:  # Only for manageable sizes
            betweenness = nx.betweenness_centrality(G)
            closeness = nx.closeness_centrality(G)
        else:
            betweenness = {node: 0 for node in G.nodes()}
            closeness = {node: 0 for node in G.nodes()}
        
        # Identify potential roles based on structural features
        for node in G.nodes():
            role_features = {
                'degree': degrees[node],
                'clustering': clustering[node],
                'betweenness': betweenness[node],
                'closeness': closeness[node]
            }
            results['node_roles'][node] = role_features
        
        # Analyze embedding space for role separation
        if ground_truth_roles is not None:
            role_separation = self._compute_role_separation(embeddings, ground_truth_roles)
            results['role_separation'] = role_separation
        
        return results
    
    def temporal_analysis(self, temporal_graphs, embeddings_func, time_steps=None):
        """
        Analyze embedding stability over temporal networks.
        
        Args:
            temporal_graphs: List of graphs at different time steps
            embeddings_func: Embedding function
            time_steps: Time step labels
        """
        if time_steps is None:
            time_steps = list(range(len(temporal_graphs)))
        
        results = {
            'time_steps': time_steps,
            'embedding_stability': [],
            'structural_drift': []
        }
        
        previous_embeddings = None
        previous_structure = None
        
        print("Running temporal analysis...")
        
        for i, G in enumerate(tqdm(temporal_graphs)):
            try:
                # Get embeddings
                embeddings = embeddings_func(G)
                
                # Compute structural features
                degrees = np.array([G.degree(node) for node in G.nodes()])
                current_structure = degrees
                
                if previous_embeddings is not None:
                    # Compute embedding stability (correlation with previous)
                    stability = self._compute_temporal_stability(previous_embeddings, embeddings)
                    results['embedding_stability'].append(stability)
                    
                    # Compute structural drift
                    if len(current_structure) == len(previous_structure):
                        drift = np.corrcoef(current_structure, previous_structure)[0, 1]
                        results['structural_drift'].append(drift)
                    else:
                        results['structural_drift'].append(np.nan)
                
                previous_embeddings = embeddings
                previous_structure = current_structure
                
            except Exception as e:
                print(f"Error at time step {time_steps[i]}: {e}")
                if previous_embeddings is not None:
                    results['embedding_stability'].append(np.nan)
                    results['structural_drift'].append(np.nan)
        
        self.results['temporal'] = results
        self._plot_temporal_results(results)
        return results
    
    # Helper methods
    def _sample_edges(self, G, sampling_rate):
        """Sample edges with given probability."""
        G_sampled = nx.Graph()
        G_sampled.add_nodes_from(G.nodes())
        
        for edge in G.edges():
            if np.random.random() < sampling_rate:
                G_sampled.add_edge(*edge)
        
        return G_sampled
    
    def _compute_pairwise_distances(self, embeddings):
        """Compute pairwise distances between embeddings."""
        from scipy.spatial.distance import pdist, squareform
        distances = pdist(embeddings, metric='euclidean')
        return squareform(distances)
    
    def _compute_preservation_score(self, original_similarity, sampled_similarity, node_mapping):
        """Compute how well structural similarity is preserved."""
        from scipy.stats import pearsonr
        
        # Extract corresponding entries
        n = len(node_mapping)
        if n != sampled_similarity.shape[0]:
            return 0
        
        orig_flat = original_similarity[np.triu_indices(n, k=1)]
        samp_flat = sampled_similarity[np.triu_indices(n, k=1)]
        
        if len(orig_flat) == 0 or len(samp_flat) == 0:
            return 0
        
        correlation, _ = pearsonr(orig_flat, samp_flat)
        return correlation if not np.isnan(correlation) else 0
    
    def _evaluate_classification(self, embeddings, labels, metric='accuracy'):
        """Evaluate classification performance."""
        if len(set(labels)) < 2:
            return 0
        
        # Standardize embeddings
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(embeddings)
        
        # Cross-validation
        clf = LogisticRegression(random_state=42, max_iter=1000)
        
        if metric == 'accuracy':
            scores = cross_val_score(clf, X_scaled, labels, cv=5, scoring='accuracy')
        elif metric == 'f1_macro':
            scores = cross_val_score(clf, X_scaled, labels, cv=5, scoring='f1_macro')
        else:
            scores = cross_val_score(clf, X_scaled, labels, cv=5)
        
        return np.mean(scores)
    
    def _compute_embedding_quality(self, G, embeddings):
        """Compute a general embedding quality metric."""
        # Simple quality metric based on degree correlation
        degrees = np.array([G.degree(node) for node in G.nodes()])
        
        # PCA to get main embedding direction
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        main_component = pca.fit_transform(embeddings).flatten()
        
        # Correlation between degree and main embedding component
        from scipy.stats import pearsonr
        correlation, _ = pearsonr(degrees, main_component)
        
        return abs(correlation) if not np.isnan(correlation) else 0
    
    def _compute_silhouette_score(self, embeddings, labels):
        """Compute silhouette score."""
        from sklearn.metrics import silhouette_score
        
        if len(set(labels)) < 2:
            return 0
        
        try:
            return silhouette_score(embeddings, labels)
        except:
            return 0
    
    def _compute_role_separation(self, embeddings, roles):
        """Compute how well different roles are separated in embedding space."""
        from sklearn.metrics import silhouette_score
        
        unique_roles = list(set(roles))
        if len(unique_roles) < 2:
            return 0
        
        try:
            return silhouette_score(embeddings, roles)
        except:
            return 0
    
    def _compute_temporal_stability(self, prev_embeddings, curr_embeddings):
        """Compute stability between temporal embeddings."""
        from scipy.stats import pearsonr
        
        if prev_embeddings.shape != curr_embeddings.shape:
            return 0
        
        # Compute correlation between flattened embeddings
        prev_flat = prev_embeddings.flatten()
        curr_flat = curr_embeddings.flatten()
        
        correlation, _ = pearsonr(prev_flat, curr_flat)
        return correlation if not np.isnan(correlation) else 0
    
    # Plotting methods
    def _plot_robustness_results(self, results):
        """Plot robustness experiment results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Structural preservation
        ax1.errorbar(results['sampling_rates'], results['structural_preservation'],
                    yerr=results['std_preservation'], marker='o', capsize=5)
        ax1.set_xlabel('Edge Sampling Rate')
        ax1.set_ylabel('Structural Preservation')
        ax1.set_title('Robustness to Edge Removal - Structural Preservation')
        ax1.grid(True, alpha=0.3)
        
        # Classification performance
        if any(score > 0 for score in results['classification_performance']):
            ax2.errorbar(results['sampling_rates'], results['classification_performance'],
                        yerr=results['std_classification'], marker='s', capsize=5, color='orange')
            ax2.set_xlabel('Edge Sampling Rate')
            ax2.set_ylabel('Classification Accuracy')
            ax2.set_title('Robustness to Edge Removal - Classification')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No classification data', ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/robustness_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_scalability_results(self, results):
        """Plot scalability experiment results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Computation time
        axes[0, 0].loglog(results['sizes'], results['computation_times'], 'o-')
        axes[0, 0].set_xlabel('Network Size')
        axes[0, 0].set_ylabel('Computation Time (s)')
        axes[0, 0].set_title('Scalability - Computation Time')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Memory usage
        axes[0, 1].loglog(results['sizes'], results['memory_usage'], 's-', color='orange')
        axes[0, 1].set_xlabel('Network Size')
        axes[0, 1].set_ylabel('Memory Usage (MB)')
        axes[0, 1].set_title('Scalability - Memory Usage')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Embedding quality
        axes[1, 0].semilogx(results['sizes'], results['embedding_quality'], '^-', color='green')
        axes[1, 0].set_xlabel('Network Size')
        axes[1, 0].set_ylabel('Embedding Quality')
        axes[1, 0].set_title('Scalability - Quality')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Efficiency (quality per time)
        efficiency = np.array(results['embedding_quality']) / np.array(results['computation_times'])
        axes[1, 1].semilogx(results['sizes'], efficiency, 'd-', color='red')
        axes[1, 1].set_xlabel('Network Size')
        axes[1, 1].set_ylabel('Efficiency (Quality/Time)')
        axes[1, 1].set_title('Scalability - Efficiency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/scalability_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_parameter_sensitivity(self, results):
        """Plot parameter sensitivity results."""
        n_params = len(results)
        fig, axes = plt.subplots(2, n_params, figsize=(5*n_params, 10))
        
        if n_params == 1:
            axes = axes.reshape(2, 1)
        
        for i, (param_name, param_results) in enumerate(results.items()):
            # Classification scores
            if any(score > 0 for score in param_results['classification_scores']):
                axes[0, i].plot(param_results['values'], param_results['classification_scores'], 'o-')
                axes[0, i].set_xlabel(param_name)
                axes[0, i].set_ylabel('Classification Score')
                axes[0, i].set_title(f'Sensitivity - {param_name} (Classification)')
                axes[0, i].grid(True, alpha=0.3)
            
            # Structural quality
            axes[1, i].plot(param_results['values'], param_results['structural_quality'], 's-', color='orange')
            axes[1, i].set_xlabel(param_name)
            axes[1, i].set_ylabel('Structural Quality')
            axes[1, i].set_title(f'Sensitivity - {param_name} (Quality)')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/parameter_sensitivity.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_cross_domain_results(self, results):
        """Plot cross-domain evaluation results."""
        datasets = results['datasets']
        metrics = list(results['metrics'].keys())
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            values = results['metrics'][metric]
            bars = axes[i].bar(datasets, values, alpha=0.8)
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_title(f'Cross-Domain Performance - {metric.capitalize()}')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                if not np.isnan(value):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/cross_domain_evaluation.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_ablation_results(self, results):
        """Plot ablation study results."""
        plt.figure(figsize=(12, 6))
        
        configs = results['configurations']
        performance = results['performance']
        
        bars = plt.bar(configs, performance, alpha=0.8)
        plt.ylabel('Performance')
        plt.title('Ablation Study Results')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, performance):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/ablation_study.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_temporal_results(self, results):
        """Plot temporal analysis results."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        time_steps = results['time_steps'][1:]  # Skip first time step
        
        # Embedding stability
        if results['embedding_stability']:
            ax1.plot(time_steps, results['embedding_stability'], 'o-', label='Embedding Stability')
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Stability (Correlation)')
            ax1.set_title('Temporal Embedding Stability')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        # Structural drift
        if results['structural_drift']:
            ax2.plot(time_steps, results['structural_drift'], 's-', color='orange', label='Structural Correlation')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Structural Correlation')
            ax2.set_title('Temporal Structural Changes')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/temporal_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, filename="experimental_results.json"):
        """Save all experimental results to file."""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        serializable_results = convert_numpy(self.results)
        
        with open(f"{self.output_dir}/{filename}", 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {self.output_dir}/{filename}")


def create_sample_experiments():
    """
    Create sample experiments with synthetic data for demonstration.
    """
    import os
    os.makedirs("experimental_results", exist_ok=True)
    
    exp = AdvancedStruc2VecExperiments()
    
    # Generate sample graph
    G = nx.barabasi_albert_graph(100, 3, seed=42)
    labels = [i % 4 for i in range(100)]  # 4 classes
    
    # Dummy embedding function
    def dummy_embeddings(graph):
        n_nodes = len(graph.nodes())
        return np.random.randn(n_nodes, 64)
    
    print("Running sample experiments...")
    
    # Scalability experiment
    def graph_generator(size):
        return nx.barabasi_albert_graph(size, 3, seed=42)
    
    scalability_results = exp.scalability_experiment(
        graph_generator, dummy_embeddings, sizes=[50, 100, 200]
    )
    
    # Cross-domain evaluation
    datasets = {
        'BA_Graph': (nx.barabasi_albert_graph(100, 3, seed=42), labels),
        'ER_Graph': (nx.erdos_renyi_graph(100, 0.1, seed=42), labels),
        'WS_Graph': (nx.watts_strogatz_graph(100, 6, 0.3, seed=42), labels)
    }
    
    cross_domain_results = exp.cross_domain_evaluation(datasets, dummy_embeddings)
    
    # Save results
    exp.save_results()
    
    print("Sample experiments completed!")


if __name__ == "__main__":
    create_sample_experiments()
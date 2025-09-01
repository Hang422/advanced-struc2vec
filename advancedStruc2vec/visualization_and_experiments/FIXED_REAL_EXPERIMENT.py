#!/usr/bin/env python3
"""
Fixed Real Algorithms Experiment
Fixed version with proper embedding handling.
"""

import sys
import os
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
import time
import warnings
warnings.filterwarnings('ignore')

# Add project paths
sys.path.append('/Users/luohang/PycharmProjects/graphlet_struc2vec/advancedStruc2vec')
sys.path.append('/Users/luohang/PycharmProjects/graphlet_struc2vec/libs')

class FixedRealExperiment:
    """Fixed version of real algorithms experiment."""
    
    def __init__(self, dataset_name='usa-airports'):
        self.dataset_name = dataset_name
        self.output_dir = 'fixed_real_results'
        os.makedirs(self.output_dir, exist_ok=True)
        self.graph = None
        self.labels = None
        
    def load_dataset(self):
        """Load dataset with proper error handling."""
        print(f"Loading {self.dataset_name} dataset...")
        
        if self.dataset_name == 'usa-airports':
            edge_file = "data/raw/data/flight/usa-airports.edgelist"
            label_file = "data/raw/data/flight/labels-usa-airports.txt"
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
        
        # Load edges
        edges = []
        with open(edge_file, 'r') as f:
            for line in f:
                u, v = map(int, line.strip().split())
                edges.append((u, v))
        
        # Load labels
        labels_dict = {}
        with open(label_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                node, label = line.strip().split()
                labels_dict[int(node)] = int(label)
        
        # Create graph
        self.graph = nx.Graph()
        self.graph.add_edges_from(edges)
        
        # Filter to labeled nodes and relabel sequentially
        labeled_nodes = set(labels_dict.keys())
        nodes_in_graph = set(self.graph.nodes())
        valid_nodes = sorted(labeled_nodes.intersection(nodes_in_graph))
        
        # Create mapping from original to sequential IDs
        node_mapping = {node: i for i, node in enumerate(valid_nodes)}
        self.graph = nx.relabel_nodes(self.graph, node_mapping)
        self.labels = np.array([labels_dict[node] for node in valid_nodes])
        
        print(f"Dataset loaded:")
        print(f"  Nodes: {len(self.graph)}")
        print(f"  Edges: {self.graph.number_of_edges()}")
        print(f"  Classes: {len(np.unique(self.labels))}")
        print(f"  Class distribution: {np.bincount(self.labels)}")
        
        return self.graph, self.labels
    
    def run_original_struc2vec(self):
        """Run original Struc2Vec with proper embedding extraction."""
        print("\\n🔄 Running Original Struc2Vec...")
        
        try:
            from GraphEmbedding.ge.models.struc2vec import Struc2Vec
            
            # Reduced parameters for speed
            params = {
                'walk_length': 10,
                'num_walks': 2,
                'workers': 1,
                'verbose': 1,
                'opt1_reduce_len': True,
                'opt2_reduce_sim_calc': True
            }
            
            model = Struc2Vec(self.graph, **params)
            
            start_time = time.time()
            model.train(embed_size=64, window_size=5, iter=1)
            training_time = time.time() - start_time
            
            # Get embeddings - handle different return formats
            embeddings_dict = model.get_embeddings()
            
            if isinstance(embeddings_dict, dict):
                # Convert dict to numpy array in correct order
                n_nodes = len(self.graph)
                embed_dim = len(next(iter(embeddings_dict.values())))
                embeddings = np.zeros((n_nodes, embed_dim))
                
                for node_id in range(n_nodes):
                    if node_id in embeddings_dict:
                        embeddings[node_id] = embeddings_dict[node_id]
                    else:
                        # Fill missing nodes with zeros
                        embeddings[node_id] = np.zeros(embed_dim)
            else:
                embeddings = np.array(embeddings_dict)
            
            print(f"✅ Original Struc2Vec completed in {training_time:.2f}s")
            print(f"   Embeddings shape: {embeddings.shape}")
            
            return embeddings, training_time
            
        except Exception as e:
            print(f"❌ Error in Original Struc2Vec: {e}")
            raise
    
    def run_enhanced_struc2vec(self):
        """Run enhanced version by adding structural features."""
        print("\\n🔄 Running Enhanced Struc2Vec...")
        
        # Get base embeddings
        base_embeddings, base_time = self.run_original_struc2vec()
        
        # Add enhancement time
        start_time = time.time()
        
        print("  Adding structural enhancements...")
        
        # Calculate additional structural features
        n_nodes = len(self.graph)
        n_features = 8  # Number of additional features
        
        enhanced_embeddings = np.zeros((n_nodes, base_embeddings.shape[1] + n_features))
        enhanced_embeddings[:, :base_embeddings.shape[1]] = base_embeddings
        
        # Feature 1: Node degree
        degrees = np.array([self.graph.degree(node) for node in range(n_nodes)])
        enhanced_embeddings[:, -8] = (degrees - degrees.mean()) / (degrees.std() + 1e-8)
        
        # Feature 2: Clustering coefficient
        clustering = np.array([nx.clustering(self.graph, node) for node in range(n_nodes)])
        enhanced_embeddings[:, -7] = (clustering - clustering.mean()) / (clustering.std() + 1e-8)
        
        # Feature 3: Degree squared (non-linear feature)
        enhanced_embeddings[:, -6] = ((degrees**2 - (degrees**2).mean()) / 
                                     ((degrees**2).std() + 1e-8))
        
        # Feature 4: Triangles count
        triangles = np.array([sum(1 for _ in nx.triangles(self.graph, node)) 
                             for node in range(n_nodes)])
        enhanced_embeddings[:, -5] = ((triangles - triangles.mean()) / 
                                     (triangles.std() + 1e-8))
        
        # Feature 5: Degree * Clustering interaction
        interaction = degrees * clustering
        enhanced_embeddings[:, -4] = ((interaction - interaction.mean()) / 
                                     (interaction.std() + 1e-8))
        
        # Feature 6: Square root of degree (another non-linear transform)
        sqrt_degrees = np.sqrt(degrees + 1)
        enhanced_embeddings[:, -3] = ((sqrt_degrees - sqrt_degrees.mean()) / 
                                     (sqrt_degrees.std() + 1e-8))
        
        # Feature 7: Log degree
        log_degrees = np.log(degrees + 1)
        enhanced_embeddings[:, -2] = ((log_degrees - log_degrees.mean()) / 
                                     (log_degrees.std() + 1e-8))
        
        # Feature 8: Normalized degree centrality
        enhanced_embeddings[:, -1] = degrees / degrees.max()
        
        enhancement_time = time.time() - start_time
        total_time = base_time + enhancement_time
        
        print(f"✅ Enhanced Struc2Vec completed in {total_time:.2f}s")
        print(f"   Base time: {base_time:.2f}s, Enhancement time: {enhancement_time:.2f}s")
        print(f"   Embeddings shape: {enhanced_embeddings.shape}")
        
        return enhanced_embeddings, total_time
    
    def evaluate_embeddings(self, embeddings_dict):
        """Evaluate embeddings with proper error handling."""
        print("\\n📊 Evaluating embeddings...")
        
        results = {}
        
        for method_name, (embeddings, train_time) in embeddings_dict.items():
            print(f"\\n  Evaluating {method_name}...")
            print(f"    Embeddings shape: {embeddings.shape}")
            print(f"    Labels shape: {self.labels.shape}")
            
            # Check dimensions match
            if embeddings.shape[0] != len(self.labels):
                print(f"    ⚠️  Dimension mismatch! Skipping...")
                continue
            
            # Check for NaN or infinite values
            if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
                print(f"    ⚠️  Found NaN/Inf values! Cleaning...")
                embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Standardize embeddings
            scaler = StandardScaler()
            try:
                X_scaled = scaler.fit_transform(embeddings)
            except Exception as e:
                print(f"    ❌ Standardization failed: {e}")
                continue
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, self.labels, test_size=0.3, 
                random_state=42, stratify=self.labels
            )
            
            # Train classifier
            clf = LogisticRegression(random_state=42, max_iter=1000)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            
            # Cross-validation
            try:
                cv_scores = cross_val_score(clf, X_scaled, self.labels, cv=5, scoring='accuracy')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            except Exception as e:
                print(f"    ⚠️  CV failed: {e}")
                cv_mean, cv_std = accuracy, 0.0
            
            results[method_name] = {
                'training_time': train_time,
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'cv_accuracy_mean': cv_mean,
                'cv_accuracy_std': cv_std,
                'embedding_dim': embeddings.shape[1]
            }
            
            print(f"    ✅ Test Accuracy: {accuracy:.4f}")
            print(f"    ✅ F1-macro: {f1_macro:.4f}")
            print(f"    ✅ CV Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
        
        return results
    
    def create_visualizations(self, embeddings_dict, results):
        """Create visualizations with error handling."""
        print("\\n🎨 Creating visualizations...")
        
        # 1. PCA comparison
        valid_embeddings = {}
        for method_name, (embeddings, _) in embeddings_dict.items():
            if embeddings.shape[0] == len(self.labels):
                valid_embeddings[method_name] = embeddings
        
        if len(valid_embeddings) > 0:
            self._create_pca_plot(valid_embeddings)
        
        # 2. Performance comparison
        self._create_performance_plot(results)
        
        # 3. Training time comparison
        self._create_timing_plot(results)
        
        print(f"  📊 Visualizations saved to {self.output_dir}/")
    
    def _create_pca_plot(self, embeddings_dict):
        """Create PCA comparison plot."""
        fig, axes = plt.subplots(1, len(embeddings_dict), figsize=(7*len(embeddings_dict), 6))
        if len(embeddings_dict) == 1:
            axes = [axes]
        
        for i, (method_name, embeddings) in enumerate(embeddings_dict.items()):
            # Apply PCA
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)
            
            # Create scatter plot
            scatter = axes[i].scatter(
                embeddings_2d[:, 0], embeddings_2d[:, 1],
                c=self.labels, cmap='tab10', alpha=0.6, s=10
            )
            axes[i].set_title(f'{method_name}\\nPCA Visualization')
            axes[i].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
            axes[i].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
            axes[i].grid(True, alpha=0.3)
            
            if i == len(embeddings_dict) - 1:
                plt.colorbar(scatter, ax=axes[i], label='Airport Region')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/pca_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_performance_plot(self, results):
        """Create performance comparison plot."""
        methods = list(results.keys())
        metrics = ['accuracy', 'f1_macro', 'cv_accuracy_mean']
        metric_names = ['Test Accuracy', 'F1-Macro', 'CV Accuracy']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            values = [results[method][metric] for method in methods]
            
            bars = axes[i].bar(methods, values, alpha=0.8, 
                              color=['skyblue', 'lightcoral'][:len(methods)])
            axes[i].set_ylabel(metric_name)
            axes[i].set_title(f'{metric_name} Comparison')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                           f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/performance_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_timing_plot(self, results):
        """Create timing comparison plot."""
        methods = list(results.keys())
        times = [results[method]['training_time'] for method in methods]
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(methods, times, alpha=0.8, color=['skyblue', 'lightcoral'][:len(methods)])
        plt.ylabel('Training Time (seconds)')
        plt.title('Training Time Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, time_val in zip(bars, times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/timing_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self, results):
        """Generate and save experiment summary."""
        print("\\n📝 Generating summary report...")
        
        report = []
        report.append("="*70)
        report.append(f"REAL ALGORITHMS EXPERIMENT - {self.dataset_name.upper()}")
        report.append("="*70)
        report.append("")
        
        # Dataset info
        report.append("DATASET INFORMATION:")
        report.append(f"  Dataset: {self.dataset_name}")
        report.append(f"  Nodes: {len(self.graph)}")
        report.append(f"  Edges: {self.graph.number_of_edges()}")
        report.append(f"  Classes: {len(np.unique(self.labels))}")
        report.append(f"  Distribution: {np.bincount(self.labels).tolist()}")
        report.append("")
        
        # Training times
        report.append("TRAINING TIME COMPARISON:")
        for method, method_results in results.items():
            report.append(f"  {method}: {method_results['training_time']:.2f} seconds")
        report.append("")
        
        # Performance results
        report.append("PERFORMANCE COMPARISON:")
        report.append("-"*30)
        
        methods = list(results.keys())
        for metric in ['accuracy', 'f1_macro', 'cv_accuracy_mean']:
            report.append(f"\\n{metric.replace('_', ' ').title()}:")
            
            # Sort methods by performance
            method_scores = [(method, results[method][metric]) for method in methods]
            method_scores.sort(key=lambda x: x[1], reverse=True)
            
            for i, (method, score) in enumerate(method_scores):
                prefix = "🏆 " if i == 0 else "   "
                report.append(f"  {prefix}{method}: {score:.4f}")
            
            if len(method_scores) > 1:
                improvement = method_scores[0][1] - method_scores[1][1]
                improvement_pct = (improvement / method_scores[1][1]) * 100
                report.append(f"      → Improvement: +{improvement:.4f} ({improvement_pct:.1f}%)")
        
        # Overall winner
        report.append("\\nOVERALL SUMMARY:")
        report.append("-"*20)
        
        # Average performance
        avg_scores = {}
        for method in methods:
            avg_score = (results[method]['accuracy'] + results[method]['f1_macro']) / 2
            avg_scores[method] = avg_score
        
        best_method = max(avg_scores, key=avg_scores.get)
        report.append(f"🏆 Best overall method: {best_method}")
        report.append(f"📊 Average performance: {avg_scores[best_method]:.4f}")
        
        # Save report
        report_text = "\\n".join(report)
        with open(f"{self.output_dir}/experiment_summary.txt", 'w') as f:
            f.write(report_text)
        
        print("\\n" + report_text)
        print(f"\\n📄 Full report saved to: {self.output_dir}/experiment_summary.txt")
    
    def run_complete_experiment(self):
        """Run the complete experiment."""
        print("="*70)
        print("🚀 FIXED REAL ALGORITHMS EXPERIMENT")
        print("="*70)
        
        # Load data
        self.load_dataset()
        
        # Run algorithms
        embeddings_dict = {}
        
        # Try to run both methods
        try:
            print("\\n" + "="*40)
            print("🔬 RUNNING ALGORITHMS")
            print("="*40)
            
            # Original Struc2Vec - run only once to avoid duplication
            original_emb, original_time = self.run_original_struc2vec()
            embeddings_dict['Original Struc2Vec'] = (original_emb, original_time)
            
            # Enhanced version - uses the same base but with added features
            print("\\n🔄 Running Enhanced Struc2Vec (with structural features)...")
            start_time = time.time()
            enhanced_emb = self._add_structural_features_to_embeddings(original_emb)
            enhancement_time = time.time() - start_time
            total_enhanced_time = original_time + enhancement_time
            
            embeddings_dict['Enhanced Struc2Vec'] = (enhanced_emb, total_enhanced_time)
            
            print(f"✅ Enhanced Struc2Vec completed in {total_enhanced_time:.2f}s")
            print(f"   (Original: {original_time:.2f}s + Enhancement: {enhancement_time:.2f}s)")
            
        except Exception as e:
            print(f"❌ Error running algorithms: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        if not embeddings_dict:
            print("❌ No algorithms completed successfully!")
            return None
        
        # Evaluate
        print("\\n" + "="*40)
        print("📊 EVALUATION")
        print("="*40)
        
        results = self.evaluate_embeddings(embeddings_dict)
        
        if not results:
            print("❌ No evaluation results obtained!")
            return None
        
        # Visualizations
        self.create_visualizations(embeddings_dict, results)
        
        # Summary report
        self.generate_summary_report(results)
        
        print("\\n" + "="*70)
        print("✅ EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"📁 Results directory: {self.output_dir}/")
        
        return results
    
    def _add_structural_features_to_embeddings(self, base_embeddings):
        """Add structural features to base embeddings."""
        print("  🔧 Adding structural features...")
        
        n_nodes = len(self.graph)
        n_additional_features = 6
        
        enhanced_embeddings = np.zeros((n_nodes, base_embeddings.shape[1] + n_additional_features))
        enhanced_embeddings[:, :base_embeddings.shape[1]] = base_embeddings
        
        # Calculate structural features
        degrees = np.array([self.graph.degree(node) for node in range(n_nodes)])
        clustering = np.array([nx.clustering(self.graph, node) for node in range(n_nodes)])
        
        # Normalize and add features
        base_idx = base_embeddings.shape[1]
        
        # Feature 1: Normalized degree
        enhanced_embeddings[:, base_idx] = (degrees - degrees.mean()) / (degrees.std() + 1e-8)
        
        # Feature 2: Normalized clustering coefficient
        enhanced_embeddings[:, base_idx + 1] = (clustering - clustering.mean()) / (clustering.std() + 1e-8)
        
        # Feature 3: Degree-clustering interaction
        interaction = degrees * clustering
        enhanced_embeddings[:, base_idx + 2] = (interaction - interaction.mean()) / (interaction.std() + 1e-8)
        
        # Feature 4: Square root of degree
        sqrt_deg = np.sqrt(degrees + 1)
        enhanced_embeddings[:, base_idx + 3] = (sqrt_deg - sqrt_deg.mean()) / (sqrt_deg.std() + 1e-8)
        
        # Feature 5: Log degree
        log_deg = np.log(degrees + 1)
        enhanced_embeddings[:, base_idx + 4] = (log_deg - log_deg.mean()) / (log_deg.std() + 1e-8)
        
        # Feature 6: Triangle count (approximate)
        triangles = np.array([len(list(nx.common_neighbors(self.graph, node, neighbor))) 
                             for node in range(min(n_nodes, 100))  # Sample to avoid timeout
                             for neighbor in list(self.graph.neighbors(node))[:1]])
        
        if len(triangles) >= n_nodes:
            triangles = triangles[:n_nodes]
        else:
            triangles = np.pad(triangles, (0, n_nodes - len(triangles)), 'constant')
        
        enhanced_embeddings[:, base_idx + 5] = (triangles - triangles.mean()) / (triangles.std() + 1e-8)
        
        print(f"    ✅ Enhanced: {base_embeddings.shape} → {enhanced_embeddings.shape}")
        
        return enhanced_embeddings


def main():
    """Main execution."""
    print("🎯 Starting Fixed Real Algorithms Experiment")
    print("📊 Dataset: USA Airports")
    print("🔬 Methods: Original Struc2Vec vs Enhanced Struc2Vec")
    print()
    
    experiment = FixedRealExperiment(dataset_name='usa-airports')
    results = experiment.run_complete_experiment()
    
    if results:
        print("\\n🎉 Experiment completed successfully!")
        print("📁 Check the 'fixed_real_results' directory for all outputs")
    else:
        print("\\n❌ Experiment failed!")
    
    return results


if __name__ == "__main__":
    results = main()
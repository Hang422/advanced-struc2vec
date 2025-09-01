#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Feature Fusion Methods
Advanced fusion techniques based on latest 2023-2024 research
"""
import sys
import numpy as np
import pickle
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Any
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add parent project path
parent_path = Path(__file__).parent.parent.parent.parent
sys.path.append(str(parent_path))

from libs.GraphEmbedding.ge.models.struc2vec import Struc2Vec

class MultiHeadAttentionFusion:
    """
    Multi-head attention fusion mechanism
    Use Transformer-style attention to learn optimal fusion weights
    """
    
    def __init__(self, num_heads=4, feature_dim=64):
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // num_heads
        
    def compute_attention_weights(self, graph_features, node_pairs):
        """Compute attention weights"""
        attention_weights = {}
        
        for pair in node_pairs:
            u, v = pair
            
            # Extract node features
            u_features = graph_features.get(u, np.zeros(self.feature_dim))
            v_features = graph_features.get(v, np.zeros(self.feature_dim))
            
            # Compute multi-head attention
            heads_weights = []
            for head in range(self.num_heads):
                start_idx = head * self.head_dim
                end_idx = (head + 1) * self.head_dim
                
                u_head = u_features[start_idx:end_idx]
                v_head = v_features[start_idx:end_idx]
                
                # Simplified attention computation
                similarity = np.dot(u_head, v_head) / (np.linalg.norm(u_head) * np.linalg.norm(v_head) + 1e-8)
                weight = 1 / (1 + np.exp(-similarity))  # sigmoid activation
                heads_weights.append(weight)
            
            # Aggregate multi-head results
            attention_weights[pair] = np.mean(heads_weights)
            
        return attention_weights
    
    def fuse_distances(self, dist1, dist2, graph_features):
        """Fuse distances using attention mechanism"""
        node_pairs = list(set(dist1.keys()) & set(dist2.keys()))
        attention_weights = self.compute_attention_weights(graph_features, node_pairs)
        
        fused_distances = {}
        for pair in node_pairs:
            alpha = attention_weights.get(pair, 0.5)
            
            fused_distances[pair] = {}
            layers1 = dist1[pair]
            layers2 = dist2[pair]
            
            for layer in set(layers1.keys()) & set(layers2.keys()):
                fused_distances[pair][layer] = alpha * layers1[layer] + (1 - alpha) * layers2[layer]
        
        return fused_distances

class HierarchicalPyramidFusion:
    """
    Hierarchical pyramid fusion
    Perform feature fusion at different scales
    """
    
    def __init__(self, pyramid_levels=3):
        self.pyramid_levels = pyramid_levels
        
    def build_feature_pyramid(self, distances_dict, max_layers):
        """Build feature pyramid"""
        pyramid = {}
        
        for level in range(self.pyramid_levels):
            scale_factor = 2 ** level
            pyramid[level] = {}
            
            for pair, layers in distances_dict.items():
                pyramid[level][pair] = {}
                
                # Aggregate layers at different scales
                for layer_idx in range(0, max_layers, scale_factor):
                    aggregated_layers = []
                    for i in range(scale_factor):
                        if layer_idx + i in layers:
                            aggregated_layers.append(layers[layer_idx + i])
                    
                    if aggregated_layers:
                        pyramid[level][pair][layer_idx] = np.mean(aggregated_layers)
        
        return pyramid
    
    def fuse_pyramid_levels(self, pyramid1, pyramid2):
        """Fuse pyramid levels"""
        fused_pyramid = {}
        
        for level in range(self.pyramid_levels):
            if level in pyramid1 and level in pyramid2:
                level_weight = 1.0 / (level + 1)  # Higher level weights are larger
                
                fused_pyramid[level] = {}
                pairs1 = set(pyramid1[level].keys())
                pairs2 = set(pyramid2[level].keys())
                
                for pair in pairs1 & pairs2:
                    fused_pyramid[level][pair] = {}
                    layers1 = pyramid1[level][pair]
                    layers2 = pyramid2[level][pair]
                    
                    for layer in set(layers1.keys()) & set(layers2.keys()):
                        fused_pyramid[level][pair][layer] = (
                            level_weight * layers1[layer] + 
                            (1 - level_weight) * layers2[layer]
                        )
        
        return fused_pyramid
    
    def flatten_pyramid(self, fused_pyramid):
        """Flatten pyramid to final distances"""
        final_distances = {}
        
        # Collect all node pairs
        all_pairs = set()
        for level_data in fused_pyramid.values():
            all_pairs.update(level_data.keys())
        
        for pair in all_pairs:
            final_distances[pair] = {}
            
            # Aggregate information from all levels
            layer_values = {}
            for level, level_data in fused_pyramid.items():
                if pair in level_data:
                    for layer, value in level_data[pair].items():
                        if layer not in layer_values:
                            layer_values[layer] = []
                        layer_values[layer].append(value)
            
            # Take average as final distance
            for layer, values in layer_values.items():
                final_distances[pair][layer] = np.mean(values)
        
        return final_distances

class SpectralAwareFusion:
    """
    Spectral-aware fusion
    Utilize spectral properties of graphs for fusion
    """
    
    def __init__(self, num_eigenvectors=10):
        self.num_eigenvectors = num_eigenvectors
    
    def compute_spectral_features(self, graph):
        """Compute spectral features of graph"""
        try:
            # Compute Laplacian matrix
            L = nx.normalized_laplacian_matrix(graph).astype(float)
            
            # Compute eigenvalues and eigenvectors
            eigenvals, eigenvecs = np.linalg.eigh(L.toarray())
            
            # Select top k eigenvectors
            selected_eigenvecs = eigenvecs[:, :self.num_eigenvectors]
            selected_eigenvals = eigenvals[:self.num_eigenvectors]
            
            return selected_eigenvals, selected_eigenvecs
            
        except Exception as e:
            print(f"Spectral computation failed: {e}")
            return None, None
    
    def compute_spectral_similarity(self, node1, node2, eigenvecs):
        """Compute spectral similarity between nodes"""
        if eigenvecs is None:
            return 0.5
        
        try:
            # Node indices in graph
            nodes = list(range(len(eigenvecs)))
            if node1 in nodes and node2 in nodes:
                vec1 = eigenvecs[node1]
                vec2 = eigenvecs[node2]
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
                return (similarity + 1) / 2  # Normalize to [0,1]
            else:
                return 0.5
        except:
            return 0.5
    
    def fuse_with_spectral_weights(self, dist1, dist2, graph):
        """Fuse distances using spectral weights"""
        eigenvals, eigenvecs = self.compute_spectral_features(graph)
        
        if eigenvecs is None:
            # Fall back to simple weighted fusion
            return self._simple_weighted_fusion(dist1, dist2, 0.5)
        
        # Create node to index mapping
        node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
        
        fused_distances = {}
        node_pairs = list(set(dist1.keys()) & set(dist2.keys()))
        
        for pair in node_pairs:
            u, v = pair
            
            # Get node indices
            u_idx = node_to_idx.get(u)
            v_idx = node_to_idx.get(v)
            
            if u_idx is not None and v_idx is not None:
                # Compute spectral similarity as fusion weight
                spectral_weight = self.compute_spectral_similarity(u_idx, v_idx, eigenvecs)
            else:
                spectral_weight = 0.5
            
            fused_distances[pair] = {}
            layers1 = dist1[pair]
            layers2 = dist2[pair]
            
            for layer in set(layers1.keys()) & set(layers2.keys()):
                fused_distances[pair][layer] = (
                    spectral_weight * layers1[layer] + 
                    (1 - spectral_weight) * layers2[layer]
                )
        
        return fused_distances
    
    def _simple_weighted_fusion(self, dist1, dist2, alpha):
        """Simple weighted fusion as fallback"""
        fused = {}
        for pair in set(dist1.keys()) & set(dist2.keys()):
            fused[pair] = {}
            for layer in set(dist1[pair].keys()) & set(dist2[pair].keys()):
                fused[pair][layer] = alpha * dist1[pair][layer] + (1 - alpha) * dist2[pair][layer]
        return fused

class CommunityAwareFusion:
    """
    Community-aware fusion
    Adjust fusion strategy based on community structure
    """
    
    def __init__(self, resolution=1.0, alpha=0.7):
        self.resolution = resolution
        self.alpha = alpha
    
    def detect_communities(self, graph):
        """Detect community structure of graph"""
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(graph, resolution=self.resolution)
            return partition
        except ImportError:
            # Use networkx community detection as alternative
            try:
                communities = nx.algorithms.community.greedy_modularity_communities(graph)
                partition = {}
                for i, community in enumerate(communities):
                    for node in community:
                        partition[node] = i
                return partition
            except:
                # If all fail, return each node as one community
                return {node: i for i, node in enumerate(graph.nodes())}
    
    def partition_pairs_by_community(self, distances, community_structure):
        """Partition node pairs by community structure"""
        intra_community = {}
        inter_community = {}
        
        for pair, layers in distances.items():
            u, v = pair
            u_community = community_structure.get(u, -1)
            v_community = community_structure.get(v, -1)
            
            if u_community == v_community and u_community != -1:
                intra_community[pair] = layers
            else:
                inter_community[pair] = layers
        
        return intra_community, inter_community
    
    def fuse_with_community_awareness(self, dist1, dist2, graph):
        """Community-aware fusion"""
        community_structure = self.detect_communities(graph)
        
        # Partition intra-community and inter-community node pairs
        intra1, inter1 = self.partition_pairs_by_community(dist1, community_structure)
        intra2, inter2 = self.partition_pairs_by_community(dist2, community_structure)
        
        # Use different fusion strategies for intra and inter-community
        fused_intra = self._fuse_intra_community(intra1, intra2, alpha=self.alpha)  # Intra-community focuses more on degree distribution
        fused_inter = self._fuse_inter_community(inter1, inter2, alpha=0.1)  # Inter-community focuses more on graphlet
        
        # Merge results
        fused_distances = {**fused_intra, **fused_inter}
        return fused_distances
    
    def _fuse_intra_community(self, dist1, dist2, alpha):
        """Intra-community fusion strategy"""
        fused = {}
        common_pairs = set(dist1.keys()) & set(dist2.keys())
        
        for pair in common_pairs:
            fused[pair] = {}
            for layer in set(dist1[pair].keys()) & set(dist2[pair].keys()):
                fused[pair][layer] = alpha * dist1[pair][layer] + (1 - alpha) * dist2[pair][layer]
        
        return fused
    
    def _fuse_inter_community(self, dist1, dist2, alpha):
        """Inter-community fusion strategy"""
        fused = {}
        common_pairs = set(dist1.keys()) & set(dist2.keys())
        
        for pair in common_pairs:
            fused[pair] = {}
            for layer in set(dist1[pair].keys()) & set(dist2[pair].keys()):
                fused[pair][layer] = alpha * dist1[pair][layer] + (1 - alpha) * dist2[pair][layer]
        
        return fused

class EnsembleFusion:
    """
    Ensemble fusion methods
    Ensemble methods combining multiple fusion strategies
    """
    
    def __init__(self, methods=None):
        if methods is None:
            self.methods = [
                ('attention', MultiHeadAttentionFusion()),
                ('spectral', SpectralAwareFusion()),
                ('community', CommunityAwareFusion())
            ]
        else:
            self.methods = methods
    
    def ensemble_fuse(self, dist1, dist2, graph):
        """Ensemble multiple fusion methods"""
        # Extract graph features for attention mechanism
        graph_features = self._extract_graph_features(graph)
        
        # Collect fusion results from all methods
        fusion_results = []
        method_weights = []
        
        for method_name, method in self.methods:
            try:
                if method_name == 'attention':
                    result = method.fuse_distances(dist1, dist2, graph_features)
                else:
                    result = method.fuse_with_community_awareness(dist1, dist2, graph) if method_name == 'community' else method.fuse_with_spectral_weights(dist1, dist2, graph)
                
                fusion_results.append(result)
                method_weights.append(1.0)  # Equal weights
                
            except Exception as e:
                print(f"Method {method_name} failed: {e}")
                continue
        
        if not fusion_results:
            # Fall back to simple fusion
            return self._simple_fusion(dist1, dist2)
        
        # Ensemble all results
        return self._ensemble_average(fusion_results, method_weights)
    
    def _extract_graph_features(self, graph):
        """Extract graph features"""
        features = {}
        
        # Compute node features
        degree_centrality = nx.degree_centrality(graph)
        clustering_coeff = nx.clustering(graph)
        
        try:
            betweenness = nx.betweenness_centrality(graph)
            closeness = nx.closeness_centrality(graph)
        except:
            betweenness = {node: 0.0 for node in graph.nodes()}
            closeness = {node: 0.0 for node in graph.nodes()}
        
        for node in graph.nodes():
            features[node] = np.array([
                degree_centrality.get(node, 0),
                clustering_coeff.get(node, 0),
                betweenness.get(node, 0),
                closeness.get(node, 0)
            ])
            
            # Pad to fixed dimensions
            while len(features[node]) < 64:
                features[node] = np.concatenate([features[node], features[node]])
            features[node] = features[node][:64]
        
        return features
    
    def _ensemble_average(self, fusion_results, weights):
        """Ensemble average multiple fusion results"""
        if not fusion_results:
            return {}
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0] * len(fusion_results)
            total_weight = len(fusion_results)
        
        normalized_weights = [w / total_weight for w in weights]
        
        # Find all common node pairs
        common_pairs = set(fusion_results[0].keys())
        for result in fusion_results[1:]:
            common_pairs &= set(result.keys())
        
        ensemble_result = {}
        for pair in common_pairs:
            ensemble_result[pair] = {}
            
            # Find all common layers
            common_layers = set(fusion_results[0][pair].keys())
            for result in fusion_results[1:]:
                if pair in result:
                    common_layers &= set(result[pair].keys())
            
            for layer in common_layers:
                weighted_sum = 0.0
                for i, result in enumerate(fusion_results):
                    if pair in result and layer in result[pair]:
                        weighted_sum += normalized_weights[i] * result[pair][layer]
                
                ensemble_result[pair][layer] = weighted_sum
        
        return ensemble_result
    
    def _simple_fusion(self, dist1, dist2, alpha=0.1):
        """Simple fusion as fallback"""
        fused = {}
        for pair in set(dist1.keys()) & set(dist2.keys()):
            fused[pair] = {}
            for layer in set(dist1[pair].keys()) & set(dist2[pair].keys()):
                fused[pair][layer] = alpha * dist1[pair][layer] + (1 - alpha) * dist2[pair][layer]
        return fused

class AdvancedFusionStruc2Vec:
    """
    Advanced fusion Struc2Vec implementation
    """
    
    def __init__(self, graph, fusion_method='ensemble', **kwargs):
        """
        Initialize advanced fusion Struc2Vec
        
        Args:
            graph: NetworkX graph
            fusion_method: fusion method ('attention', 'pyramid', 'spectral', 'community', 'ensemble')
            **kwargs: other parameters
        """
        self.graph = graph
        self.fusion_method = fusion_method
        self.fusion_params = kwargs.get('fusion_params', {})
        
        # Create fuser
        if fusion_method == 'attention':
            self.fuser = MultiHeadAttentionFusion(**self.fusion_params)
        elif fusion_method == 'pyramid':
            self.fuser = HierarchicalPyramidFusion(**self.fusion_params)
        elif fusion_method == 'spectral':
            self.fuser = SpectralAwareFusion(**self.fusion_params)
        elif fusion_method == 'community':
            self.fuser = CommunityAwareFusion(**self.fusion_params)
        elif fusion_method == 'ensemble':
            self.fuser = EnsembleFusion(**self.fusion_params)
        else:
            raise ValueError(f"未知的fusion method: {fusion_method}")
        
        # Generate advanced fusion distances
        self._generate_advanced_fused_distances(graph, **kwargs)
        
        # Create modified Struc2Vec instance
        self.model = Struc2Vec(
            graph,
            walk_length=kwargs.get('walk_length', 40),
            num_walks=kwargs.get('num_walks', 8),
            workers=kwargs.get('workers', 1),
            verbose=kwargs.get('verbose', 0),
            opt1_reduce_len=kwargs.get('opt1_reduce_len', True),
            opt2_reduce_sim_calc=kwargs.get('opt2_reduce_sim_calc', True)
        )
        
        # Replace distance files
        self._replace_structural_distances()
    
    def _replace_structural_distances(self):
        """Replace structural distance files"""
        import pickle
        
        # Load fused distances
        with open(self.fused_distance_file, 'rb') as f:
            fused_distances = pickle.load(f)
        
        # Set fused distances into model
        self.model.structural_distance = fused_distances
        
        # Create temporary file to store distances
        temp_dist_file = Path(self.model.temp_path) / "structural_dist.pkl"
        temp_dist_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_dist_file, 'wb') as f:
            pickle.dump(fused_distances, f)
        
        print(f"🔄 Distance file replacement completed: {temp_dist_file}")
    
    def train(self, embed_size=64, window_size=5, iter=3, **kwargs):
        """Train model"""
        return self.model.train(
            embed_size=embed_size,
            window_size=window_size,
            iter=iter,
            **kwargs
        )
    
    def get_embeddings(self):
        """Get embeddings"""
        return self.model.get_embeddings()
    
    def _generate_advanced_fused_distances(self, graph, **kwargs):
        """Generate advanced fusion distances"""
        from src.algorithms.graphlet_based.compute_edges_improved import generate_improved_structural_distance
        import tempfile
        import shutil
        import time
        
        # 创建临时目录
        temp_dir = Path(tempfile.mkdtemp())
        output_dir = Path(__file__).parent.parent.parent / 'output' / 'distances'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save temporary graph file
            temp_graph_file = temp_dir / 'temp_graph.edgelist'
            nx.write_edgelist(graph, str(temp_graph_file), data=False)
            
            # Generate original distances (based on degree sequence)
            original_temp = Struc2Vec(graph, walk_length=10, num_walks=1, workers=1, verbose=0)
            original_dist_file = temp_dir / 'original_dist.pkl'
            time.sleep(1)  # Ensure file generation
            if Path(original_temp.temp_path + "/structural_dist.pkl").exists():
                shutil.copy(original_temp.temp_path + "/structural_dist.pkl", original_dist_file)
                shutil.rmtree(original_temp.temp_path)
            
            # Generate graphlet distances
            graphlet_dist_file = temp_dir / 'graphlet_dist.pkl'
            generate_improved_structural_distance(
                str(temp_graph_file),
                str(graphlet_dist_file),
                max_layer=kwargs.get('max_layer', 3),
                distance_method=kwargs.get('distance_method', 'frobenius'),
                use_orbit_selection=kwargs.get('use_orbit_selection', False)
            )
            
            # Load distance files
            with open(original_dist_file, 'rb') as f:
                original_distances = pickle.load(f)
            with open(graphlet_dist_file, 'rb') as f:
                graphlet_distances = pickle.load(f)
            
            # Apply advanced fusion
            if self.fusion_method == 'pyramid':
                # Special processing for pyramid fusion
                pyramid1 = self.fuser.build_feature_pyramid(original_distances, kwargs.get('max_layer', 3))
                pyramid2 = self.fuser.build_feature_pyramid(graphlet_distances, kwargs.get('max_layer', 3))
                fused_pyramid = self.fuser.fuse_pyramid_levels(pyramid1, pyramid2)
                fused_distances = self.fuser.flatten_pyramid(fused_pyramid)
            elif self.fusion_method == 'attention':
                # Extract graph features
                graph_features = self.fuser._extract_graph_features(graph) if hasattr(self.fuser, '_extract_graph_features') else self._extract_simple_features(graph)
                fused_distances = self.fuser.fuse_distances(original_distances, graphlet_distances, graph_features)
            elif self.fusion_method == 'ensemble':
                fused_distances = self.fuser.ensemble_fuse(original_distances, graphlet_distances, graph)
            else:
                # Other methods
                if hasattr(self.fuser, 'fuse_with_spectral_weights'):
                    fused_distances = self.fuser.fuse_with_spectral_weights(original_distances, graphlet_distances, graph)
                elif hasattr(self.fuser, 'fuse_with_community_awareness'):
                    fused_distances = self.fuser.fuse_with_community_awareness(original_distances, graphlet_distances, graph)
                else:
                    # Fall back to simple fusion
                    fused_distances = self._simple_fusion(original_distances, graphlet_distances)
            
            # Save fused distances
            self.fused_distance_file = output_dir / f'advanced_fused_{self.fusion_method}_{id(graph)}.pkl'
            with open(self.fused_distance_file, 'wb') as f:
                pickle.dump(fused_distances, f)
            
            print(f"✅ Advanced fusion distance file generation completed: {self.fused_distance_file}")
            
        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir)
    
    def _extract_simple_features(self, graph):
        """Extract simple graph features"""
        features = {}
        degree_centrality = nx.degree_centrality(graph)
        clustering_coeff = nx.clustering(graph)
        
        for node in graph.nodes():
            features[node] = np.array([
                degree_centrality.get(node, 0),
                clustering_coeff.get(node, 0),
                0.0, 0.0  # Placeholder
            ])
            # Extend to 64 dimensions
            features[node] = np.tile(features[node], 16)
        
        return features
    
    def _simple_fusion(self, dist1, dist2, alpha=0.1):
        """Simple fusion as fallback"""
        fused = {}
        for pair in set(dist1.keys()) & set(dist2.keys()):
            fused[pair] = {}
            for layer in set(dist1[pair].keys()) & set(dist2[pair].keys()):
                fused[pair][layer] = alpha * dist1[pair][layer] + (1 - alpha) * dist2[pair][layer]
        return fused
    
    def get_method_name(self):
        """Get method name"""
        return f"Advanced Fusion Struc2Vec ({self.fusion_method})"


class PureGraphletStruc2Vec:
    """
    Pure Graphlet-based Struc2Vec implementation
    Uses only Graphlet distances, completely replacing original degree-based distances
    """
    
    def __init__(self, graph, **kwargs):
        """
        Initialize pure Graphlet Struc2Vec
        
        Args:
            graph: NetworkX graph
            **kwargs: other parameters
        """
        self.graph = graph
        self.embeddings = None
        
        # Validate parameters
        self._validate_parameters(graph, **kwargs)
        
        # Generate pure graphlet distances
        self._generate_pure_graphlet_distances(graph, **kwargs)
        
        # Create Struc2Vec instance with graphlet distances
        self.model = Struc2Vec(
            graph,
            walk_length=kwargs.get('walk_length', 40),
            num_walks=kwargs.get('num_walks', 8),
            workers=kwargs.get('workers', 1),
            verbose=kwargs.get('verbose', 0),
            opt1_reduce_len=kwargs.get('opt1_reduce_len', True),
            opt2_reduce_sim_calc=kwargs.get('opt2_reduce_sim_calc', True)
        )
        
        # Replace structural distances with pure graphlet distances
        self._replace_with_graphlet_distances()
    
    def _generate_pure_graphlet_distances(self, graph, **kwargs):
        """Generate pure graphlet distances"""
        from src.algorithms.graphlet_based.compute_edges_improved import generate_improved_structural_distance
        import tempfile
        import time
        
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        output_dir = Path(__file__).parent.parent.parent / 'output' / 'distances'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save temporary graph file
            temp_graph_file = temp_dir / 'temp_graph.edgelist'
            nx.write_edgelist(graph, str(temp_graph_file), data=False)
            print(f"📂 Loading graph: {temp_graph_file}")
            
            # Generate graphlet distances
            graphlet_dist_file = temp_dir / 'graphlet_dist.pkl'
            generate_improved_structural_distance(
                str(temp_graph_file),
                str(graphlet_dist_file),
                k=kwargs.get('k', 5),
                max_layer=kwargs.get('max_layer', 3),
                distance_method=kwargs.get('distance_method', 'frobenius'),
                use_orbit_selection=kwargs.get('use_orbit_selection', False),
                top_k_orbits=kwargs.get('top_k_orbits', 40)
            )
            
            # Copy to final location
            self.graphlet_distance_file = output_dir / f'pure_graphlet_{id(graph)}.pkl'
            with open(graphlet_dist_file, 'rb') as f:
                graphlet_distances = pickle.load(f)
            
            with open(self.graphlet_distance_file, 'wb') as f:
                pickle.dump(graphlet_distances, f)
            
            print(f"✅ Pure graphlet distance file generation completed: {self.graphlet_distance_file}")
            
        finally:
            # Clean up temporary files
            import shutil
            shutil.rmtree(temp_dir)
    
    def _replace_with_graphlet_distances(self):
        """Replace Struc2Vec distances with pure graphlet distances"""
        import pickle
        
        # Load graphlet distances
        with open(self.graphlet_distance_file, 'rb') as f:
            graphlet_distances = pickle.load(f)
        
        # Replace structural distances in the model
        self.model.structural_distance = graphlet_distances
        
        # Create temporary file for Struc2Vec
        temp_dist_file = Path(self.model.temp_path) / "structural_dist.pkl"
        temp_dist_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_dist_file, 'wb') as f:
            pickle.dump(graphlet_distances, f)
        
        print(f"🔄 Distance file replacement completed: {temp_dist_file}")
    
    def train(self, **kwargs):
        """Train pure graphlet Struc2Vec model"""
        print("🚀 Training Pure Graphlet Struc2Vec...")
        
        self.model.train(
            embed_size=kwargs.get('embed_size', 64),
            window_size=kwargs.get('window_size', 5),
            workers=kwargs.get('workers', 1),
            iter=kwargs.get('iter', 3)
        )
        
        # Get embeddings
        self.embeddings = self.model.get_embeddings()
        print("✅ Pure Graphlet Struc2Vec training completed")
    
    def get_embeddings(self):
        """Get node embeddings"""
        if self.embeddings is None:
            raise ValueError("Model must be trained first")
        return self.embeddings
    
    def get_method_name(self):
        """Get method name"""
        return "Pure Graphlet Struc2Vec"
    
    def get_distance_file(self):
        """Get distance file path"""
        return str(self.graphlet_distance_file)
    
    def _validate_parameters(self, graph, **kwargs):
        """Validate parameters for Pure Graphlet method"""
        # Validate graph size
        if graph.number_of_nodes() < 4:
            raise ValueError(f"Graph too small ({graph.number_of_nodes()} nodes). Minimum 4 nodes required for k=4 graphlets.")
        
        # Validate k parameter
        k = kwargs.get('k', 4)
        if k not in [4, 5]:
            raise ValueError(f"Invalid graphlet size k={k}. ORCA only supports k=4 or k=5.")
        
        # Validate graph connectivity for small graphs
        if graph.number_of_nodes() <= 10:
            # For very small graphs, check if they have sufficient connectivity
            if graph.number_of_edges() == 0:
                raise ValueError("Graph has no edges. Cannot compute graphlet features for isolated nodes.")
            
            # Check if graph is too sparse for meaningful graphlets
            min_edges_needed = max(3, graph.number_of_nodes() - 2)  # At least a tree + 1 edge
            if graph.number_of_edges() < min_edges_needed:
                print(f"⚠️ Warning: Graph may be too sparse ({graph.number_of_edges()} edges) for meaningful graphlet analysis.")
        
        # Validate other parameters
        max_layer = kwargs.get('max_layer', 3)
        if max_layer < 1:
            raise ValueError(f"max_layer must be >= 1, got {max_layer}")
        
        distance_method = kwargs.get('distance_method', 'frobenius')
        if distance_method not in ['frobenius', 'simple']:
            raise ValueError(f"Invalid distance_method '{distance_method}'. Must be 'frobenius' or 'simple'.")
        
        print(f"✅ Parameter validation passed: k={k}, max_layer={max_layer}, distance_method='{distance_method}'")


if __name__ == "__main__":
    # Test advanced fusion methods and pure graphlet method
    print("Advanced fusion methods module loaded successfully")
    print("Supported fusion methods: attention, pyramid, spectral, community, ensemble")
    print("Supported pure method: pure-graphlet")
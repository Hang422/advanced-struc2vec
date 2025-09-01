#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved graphlet-enhanced struc2vec distance computation module
Main improvements:
1. Better GDV preprocessing strategies
2. Multiple distance metric methods
3. Feature selection and adaptive weights
4. More flexible fusion strategies
"""
import os
import sys
import json
import pickle
import subprocess
import numpy as np
import networkx as nx
from collections import deque
from typing import Dict, Tuple, List, Optional
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
from scipy.spatial.distance import cosine, euclidean


def _find_orca_executable():
    # advancedStruc2vec 目录是当前项目根目录
    current_root = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir, os.pardir))
    # 真正的项目根目录在上一级
    project_root = os.path.dirname(current_root)
    
    for name in ("orca", "orca.exe"):
        # 先尝试项目根目录下的 libs/orca/ 目录
        p = os.path.join(project_root, "libs", "orca", name)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
        
        # 备用：当前目录内部 libs/orca/ 目录
        p = os.path.join(current_root, "libs", "orca", name)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    
    from shutil import which
    for name in ("orca", "orca.exe"):
        p = which(name)
        if p:
            return p
    
    raise FileNotFoundError("Cannot find orca binary")


def preprocess_edgelist(input_path: str, output_path: str) -> tuple[dict, dict]:
    """将原始 .edgelist 文件转换为 ORCA 可读格式"""
    G = nx.read_edgelist(input_path, nodetype=int)
    mapping = {node: i for i, node in enumerate(sorted(G.nodes()))}
    reverse = {i: node for node, i in mapping.items()}
    H = nx.relabel_nodes(G, mapping)
    H.remove_edges_from(nx.selfloop_edges(H))
    n = H.number_of_nodes()
    e = H.number_of_edges()
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"{n} {e}\n")
        for u, v in H.edges():
            f.write(f"{u} {v}\n")
    return mapping, reverse


def compute_node_gdv(edgelist_path: str, k: int = 5, orca_bin: str = None, mode: str = 'node') -> dict[int, np.ndarray]:
    """Call ORCA to compute node graphlet degree vectors"""
    if orca_bin is None:
        orca_bin = _find_orca_executable()
    out_path = f'{edgelist_path}_GDV.out'
    cmd = [orca_bin, mode, str(k), edgelist_path, out_path]
    subprocess.run(cmd, check=True)
    gdv = {}
    with open(out_path, 'r') as f:
        for idx, line in enumerate(f):
            gdv[idx] = np.fromstring(line, dtype=int, sep=' ')
    return gdv


class ImprovedGDVPreprocessor:
    """Improved GDV preprocessor"""
    def __init__(self, normalization_method: str = 'adaptive'):
        self.normalization_method = normalization_method
        self.scaler = StandardScaler()
        self.orbit_importance = None
        
    def fit_transform(self, gdv: np.ndarray) -> np.ndarray:
        """
        Intelligent preprocessing of GDV
        - adaptive: Adaptively select normalization method based on orbit distribution
        - zscore: 标准化
        - log: log 变换
        - sqrt: 平方根变换
        """
        n_nodes, n_orbits = gdv.shape
        processed_gdv = np.zeros_like(gdv, dtype=float)
        
        if self.normalization_method == 'adaptive':
            # Process each orbit separately
            for i in range(n_orbits):
                orbit_values = gdv[:, i]
                
                # Compute distribution characteristics
                sparsity = np.mean(orbit_values == 0)
                if sparsity > 0.9:  # Very sparse orbit
                    # Use binarization
                    processed_gdv[:, i] = (orbit_values > 0).astype(float)
                elif np.max(orbit_values) > 100:  # Orbit with large value range
                    # Use log transformation
                    processed_gdv[:, i] = np.log1p(orbit_values)
                else:
                    # 使用 z-score 标准化
                    if np.std(orbit_values) > 0:
                        processed_gdv[:, i] = (orbit_values - np.mean(orbit_values)) / np.std(orbit_values)
                    else:
                        processed_gdv[:, i] = 0
                        
        elif self.normalization_method == 'zscore':
            processed_gdv = self.scaler.fit_transform(gdv)
            
        elif self.normalization_method == 'log':
            processed_gdv = np.log1p(gdv)
            
        elif self.normalization_method == 'sqrt':
            processed_gdv = np.sqrt(gdv)
            
        # Compute orbit importance (based on variance)
        self.orbit_importance = np.var(processed_gdv, axis=0)
        self.orbit_importance = self.orbit_importance / np.sum(self.orbit_importance)
        
        return processed_gdv
    
    def select_important_orbits(self, gdv: np.ndarray, top_k: int = 40) -> Tuple[np.ndarray, List[int]]:
        """Select the most important k orbits"""
        if self.orbit_importance is None:
            raise ValueError("Need to call fit_transform first")
            
        important_indices = np.argsort(self.orbit_importance)[-top_k:]
        return gdv[:, important_indices], important_indices


def enhanced_graphlet_correlation(M: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Enhanced correlation matrix computation with weighting support"""
    n, z = M.shape
    
    if weights is not None:
        # Weighted average
        weighted_mean = np.average(M, axis=0, weights=weights)
        X = M - weighted_mean
        # Weighted covariance
        cov = np.zeros((z, z))
        for i in range(n):
            cov += weights[i] * np.outer(X[i], X[i])
        cov /= np.sum(weights)
    else:
        X = M - M.mean(axis=0, keepdims=True)
        cov = (X.T @ X) / (n - 1) if n > 1 else np.zeros((z, z), dtype=float)
    
    diag = np.diag(cov)
    std = np.sqrt(np.maximum(diag, 0))
    std_safe = std.copy()
    std_safe[std_safe == 0] = 1.0
    
    C = cov / std_safe[:, None] / std_safe[None, :]
    zero_std = (std == 0)
    C[zero_std, :] = 0.0
    C[:, zero_std] = 0.0
    
    return C


class MultiMetricDistance:
    """Multi-metric distance calculator"""
    
    @staticmethod
    def matrix_distance(C1: np.ndarray, C2: np.ndarray, method: str = 'combined') -> float:
        """
        Compute distance between two correlation matrices
        method: 'frobenius', 'eigenvalue', 'trace', 'combined'
        """
        if method == 'frobenius':
            return np.linalg.norm(C1 - C2, 'fro')
            
        elif method == 'eigenvalue':
            # Use distance of top k eigenvalues
            k = min(5, C1.shape[0])
            try:
                eigvals1 = np.sort(np.linalg.eigvalsh(C1))[-k:]
                eigvals2 = np.sort(np.linalg.eigvalsh(C2))[-k:]
                return np.linalg.norm(eigvals1 - eigvals2)
            except:
                return np.linalg.norm(C1 - C2, 'fro')
                
        elif method == 'trace':
            return abs(np.trace(C1) - np.trace(C2))
            
        elif method == 'combined':
            # 组合多种距离度量
            distances = []
            
            # Frobenius 范数
            distances.append(np.linalg.norm(C1 - C2, 'fro'))
            
            # 迹差异
            distances.append(abs(np.trace(C1) - np.trace(C2)))
            
            # 特征值距离（如果矩阵不太小）
            if C1.shape[0] >= 3:
                try:
                    k = min(3, C1.shape[0])
                    eigvals1 = np.sort(np.linalg.eigvalsh(C1))[-k:]
                    eigvals2 = np.sort(np.linalg.eigvalsh(C2))[-k:]
                    distances.append(np.linalg.norm(eigvals1 - eigvals2))
                except:
                    distances.append(0)
            
            # Weighted combination
            weights = [0.5, 0.3, 0.2] if len(distances) == 3 else [0.7, 0.3]
            return np.sum([w * d for w, d in zip(weights, distances)])
    
    @staticmethod
    def vector_distance(v1: np.ndarray, v2: np.ndarray, method: str = 'weighted_l1') -> float:
        """Compute distance between two vectors"""
        if method == 'l1':
            return np.linalg.norm(v1 - v2, ord=1)
        elif method == 'l2':
            return np.linalg.norm(v1 - v2, ord=2)
        elif method == 'cosine':
            return cosine(v1, v2) if np.any(v1) and np.any(v2) else 0
        elif method == 'weighted_l1':
            # Use different weights for different dimensions
            weights = 1 / (1 + np.arange(len(v1)))
            return np.sum(weights * np.abs(v1 - v2))


def compute_graphlet_distance_improved(
    graph: nx.Graph,
    node_gdv: np.ndarray,
    max_layer: int = 5,
    distance_method: str = 'combined',
    use_orbit_selection: bool = True,
    top_k_orbits: int = 40
) -> Dict[Tuple[int, int], Dict[int, float]]:
    """
    Improved graphlet distance computation
    - Support multiple distance metrics
    - Support orbit selection
    - More flexible hierarchical aggregation
    """
    # Preprocess GDV
    preprocessor = ImprovedGDVPreprocessor('adaptive')
    processed_gdv = preprocessor.fit_transform(node_gdv)
    
    # Select important orbits
    if use_orbit_selection:
        processed_gdv, selected_orbits = preprocessor.select_important_orbits(
            processed_gdv, top_k_orbits
        )
    
    nodes = sorted(graph.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    
    # Compute features for each node at each layer
    layer_features = [{} for _ in range(n)]
    metric_calc = MultiMetricDistance()
    
    for ui, u in enumerate(nodes):
        visited = {u}
        q = deque([(u, 0)])
        layers = {0: [u]}
        
        # BFS Get nodes at each layer
        while q:
            v, d = q.popleft()
            if d >= max_layer:
                continue
            for w in graph[v]:
                if w not in visited:
                    visited.add(w)
                    q.append((w, d + 1))
                    layers.setdefault(d + 1, []).append(w)
        
        # Compute features for each layer
        for ℓ in range(max_layer + 1):
            members = layers.get(ℓ, [])
            if len(members) == 0:
                layer_features[ui][ℓ] = {
                    'corr_matrix': np.zeros((processed_gdv.shape[1], processed_gdv.shape[1])),
                    'mean_vector': np.zeros(processed_gdv.shape[1]),
                    'std_vector': np.zeros(processed_gdv.shape[1]),
                    'size': 0
                }
            else:
                M = processed_gdv[[idx[w] for w in members], :]
                
                # Compute node importance weights (based on degree)
                degrees = np.array([graph.degree[w] for w in members])
                weights = degrees / np.sum(degrees) if np.sum(degrees) > 0 else None
                
                layer_features[ui][ℓ] = {
                    'corr_matrix': enhanced_graphlet_correlation(M, weights),
                    'mean_vector': np.average(M, axis=0, weights=weights) if weights is not None else np.mean(M, axis=0),
                    'std_vector': np.std(M, axis=0),
                    'size': len(members)
                }
    
    # Compute pairwise node distances
    distances = {}
    for i in range(n):
        for j in range(i + 1, n):
            layer_distances = {}
            
            for ℓ in range(max_layer + 1):
                feat_i = layer_features[i][ℓ]
                feat_j = layer_features[j][ℓ]
                
                # 组合多种距离
                if distance_method == 'combined':
                    # 相关性矩阵距离
                    corr_dist = metric_calc.matrix_distance(
                        feat_i['corr_matrix'], 
                        feat_j['corr_matrix'], 
                        'combined'
                    )
                    
                    # 均值向量距离
                    mean_dist = metric_calc.vector_distance(
                        feat_i['mean_vector'],
                        feat_j['mean_vector'],
                        'weighted_l1'
                    )
                    
                    # 标准差向量距离
                    std_dist = metric_calc.vector_distance(
                        feat_i['std_vector'],
                        feat_j['std_vector'],
                        'l2'
                    )
                    
                    # 大小差异
                    size_diff = abs(feat_i['size'] - feat_j['size']) / max(feat_i['size'], feat_j['size'], 1)
                    
                    # Weighted combination，近层权重更高
                    layer_weight = np.exp(-ℓ / 2)
                    layer_distances[ℓ] = layer_weight * (
                        0.4 * corr_dist + 
                        0.3 * mean_dist + 
                        0.2 * std_dist + 
                        0.1 * size_diff
                    )
                else:
                    # 使用单一距离度量
                    layer_distances[ℓ] = metric_calc.matrix_distance(
                        feat_i['corr_matrix'],
                        feat_j['corr_matrix'],
                        distance_method
                    )
            
            # 层次聚合（使用累积和，但保留原始值）
            cum_distances = {}
            cum = 0.0
            for ℓ in range(max_layer + 1):
                if ℓ in layer_distances and layer_distances[ℓ] is not None:
                    cum += layer_distances[ℓ]
                cum_distances[ℓ] = cum
                
            distances[(nodes[i], nodes[j])] = cum_distances
    
    return distances


def adaptive_fusion(
    graphlet_dist: Dict,
    degree_dist: Dict,
    graph: nx.Graph
) -> Dict:
    """
    Adaptive fusion of graphlet and degree sequence distances
    Dynamically adjust weights based on node properties
    """
    fused = {}
    
    # Compute global graph properties
    clustering = nx.average_clustering(graph)
    density = nx.density(graph)
    
    for pair in set(graphlet_dist.keys()).intersection(degree_dist.keys()):
        u, v = pair
        
        # Adjust weights based on local node properties
        u_clustering = nx.clustering(graph, u)
        v_clustering = nx.clustering(graph, v)
        avg_clustering = (u_clustering + v_clustering) / 2
        
        # For node pairs with high clustering coefficient, graphlet weight is higher
        if avg_clustering > clustering:
            alpha = 0.7  # graphlet 权重
        else:
            alpha = 0.5
        
        layers_g = graphlet_dist[pair]
        layers_d = degree_dist[pair]
        fused[pair] = {}
        
        for l in set(layers_g.keys()).intersection(layers_d.keys()):
            fused[pair][l] = alpha * layers_g[l] + (1 - alpha) * layers_d[l]
    
    return fused


def generate_improved_structural_distance(
    edgelist_path: str,
    output_path: str,
    k: int = 5,
    max_layer: int = 5,
    distance_method: str = 'combined',
    use_orbit_selection: bool = True,
    top_k_orbits: int = 40
):
    """Generate improved structural distance file"""
    print(f"📂 加载图：{edgelist_path}")
    
    # Prepare ORCA input
    orca_input = f"{edgelist_path}.in"
    mapping, reverse = preprocess_edgelist(edgelist_path, orca_input)
    
    print("🚀 Call ORCA to compute node GDV ...")
    gdv_dict = compute_node_gdv(orca_input, k=k)
    node_gdv = np.array([gdv_dict[i] for i in range(len(mapping))])
    
    # Load graph
    G = nx.read_edgelist(edgelist_path, nodetype=int)
    G = nx.relabel_nodes(G, mapping)
    
    print("🧠 Computing改进的结构距离...")
    raw_dist = compute_graphlet_distance_improved(
        G, node_gdv, 
        max_layer=max_layer,
        distance_method=distance_method,
        use_orbit_selection=use_orbit_selection,
        top_k_orbits=top_k_orbits
    )
    
    print("🔁 Restore original node numbering...")
    final_dist = {}
    for (u, v), layer_d in raw_dist.items():
        final_dist[(reverse[u], reverse[v])] = layer_d
    
    print(f"💾 Saving到：{output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(final_dist, f)
    
    # Clean up temporary files
    os.remove(orca_input)
    print("✅ Completed！")


if __name__ == '__main__':
    # Test improved methods
    generate_improved_structural_distance(
        "../data/flight/brazil-airports.edgelist",
        "output/structural_dist_improved.pkl",
        max_layer=5,
        distance_method='combined',
        use_orbit_selection=True,
        top_k_orbits=40
    )
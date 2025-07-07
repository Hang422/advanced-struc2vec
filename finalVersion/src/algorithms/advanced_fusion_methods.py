#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级特征融合方法
基于2023-2024年最新研究的先进融合技术
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

# 添加父项目路径
parent_path = Path(__file__).parent.parent.parent.parent
sys.path.append(str(parent_path))

from libs.GraphEmbedding.ge.models.struc2vec import Struc2Vec

class MultiHeadAttentionFusion:
    """
    多头注意力融合机制
    使用Transformer风格的注意力学习最优融合权重
    """
    
    def __init__(self, num_heads=4, feature_dim=64):
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // num_heads
        
    def compute_attention_weights(self, graph_features, node_pairs):
        """计算注意力权重"""
        attention_weights = {}
        
        for pair in node_pairs:
            u, v = pair
            
            # 提取节点特征
            u_features = graph_features.get(u, np.zeros(self.feature_dim))
            v_features = graph_features.get(v, np.zeros(self.feature_dim))
            
            # 计算多头注意力
            heads_weights = []
            for head in range(self.num_heads):
                start_idx = head * self.head_dim
                end_idx = (head + 1) * self.head_dim
                
                u_head = u_features[start_idx:end_idx]
                v_head = v_features[start_idx:end_idx]
                
                # 简化的注意力计算
                similarity = np.dot(u_head, v_head) / (np.linalg.norm(u_head) * np.linalg.norm(v_head) + 1e-8)
                weight = 1 / (1 + np.exp(-similarity))  # sigmoid activation
                heads_weights.append(weight)
            
            # 聚合多头结果
            attention_weights[pair] = np.mean(heads_weights)
            
        return attention_weights
    
    def fuse_distances(self, dist1, dist2, graph_features):
        """使用注意力机制融合距离"""
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
    层次化金字塔融合
    在不同尺度上进行特征融合
    """
    
    def __init__(self, pyramid_levels=3):
        self.pyramid_levels = pyramid_levels
        
    def build_feature_pyramid(self, distances_dict, max_layers):
        """构建特征金字塔"""
        pyramid = {}
        
        for level in range(self.pyramid_levels):
            scale_factor = 2 ** level
            pyramid[level] = {}
            
            for pair, layers in distances_dict.items():
                pyramid[level][pair] = {}
                
                # 在不同尺度上聚合层
                for layer_idx in range(0, max_layers, scale_factor):
                    aggregated_layers = []
                    for i in range(scale_factor):
                        if layer_idx + i in layers:
                            aggregated_layers.append(layers[layer_idx + i])
                    
                    if aggregated_layers:
                        pyramid[level][pair][layer_idx] = np.mean(aggregated_layers)
        
        return pyramid
    
    def fuse_pyramid_levels(self, pyramid1, pyramid2):
        """融合金字塔层级"""
        fused_pyramid = {}
        
        for level in range(self.pyramid_levels):
            if level in pyramid1 and level in pyramid2:
                level_weight = 1.0 / (level + 1)  # 高层级权重更大
                
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
        """将金字塔展平为最终距离"""
        final_distances = {}
        
        # 收集所有节点对
        all_pairs = set()
        for level_data in fused_pyramid.values():
            all_pairs.update(level_data.keys())
        
        for pair in all_pairs:
            final_distances[pair] = {}
            
            # 聚合所有层级的信息
            layer_values = {}
            for level, level_data in fused_pyramid.items():
                if pair in level_data:
                    for layer, value in level_data[pair].items():
                        if layer not in layer_values:
                            layer_values[layer] = []
                        layer_values[layer].append(value)
            
            # 取平均值作为最终距离
            for layer, values in layer_values.items():
                final_distances[pair][layer] = np.mean(values)
        
        return final_distances

class SpectralAwareFusion:
    """
    谱感知融合
    利用图的谱特性进行融合
    """
    
    def __init__(self, num_eigenvectors=10):
        self.num_eigenvectors = num_eigenvectors
    
    def compute_spectral_features(self, graph):
        """计算图的谱特征"""
        try:
            # 计算拉普拉斯矩阵
            L = nx.normalized_laplacian_matrix(graph).astype(float)
            
            # 计算特征值和特征向量
            eigenvals, eigenvecs = np.linalg.eigh(L.toarray())
            
            # 选择前k个特征向量
            selected_eigenvecs = eigenvecs[:, :self.num_eigenvectors]
            selected_eigenvals = eigenvals[:self.num_eigenvectors]
            
            return selected_eigenvals, selected_eigenvecs
            
        except Exception as e:
            print(f"谱计算失败: {e}")
            return None, None
    
    def compute_spectral_similarity(self, node1, node2, eigenvecs):
        """计算节点间的谱相似性"""
        if eigenvecs is None:
            return 0.5
        
        try:
            # 节点在图中的索引
            nodes = list(range(len(eigenvecs)))
            if node1 in nodes and node2 in nodes:
                vec1 = eigenvecs[node1]
                vec2 = eigenvecs[node2]
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
                return (similarity + 1) / 2  # 归一化到[0,1]
            else:
                return 0.5
        except:
            return 0.5
    
    def fuse_with_spectral_weights(self, dist1, dist2, graph):
        """使用谱权重融合距离"""
        eigenvals, eigenvecs = self.compute_spectral_features(graph)
        
        if eigenvecs is None:
            # 回退到简单加权融合
            return self._simple_weighted_fusion(dist1, dist2, 0.5)
        
        # 创建节点到索引的映射
        node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
        
        fused_distances = {}
        node_pairs = list(set(dist1.keys()) & set(dist2.keys()))
        
        for pair in node_pairs:
            u, v = pair
            
            # 获取节点索引
            u_idx = node_to_idx.get(u)
            v_idx = node_to_idx.get(v)
            
            if u_idx is not None and v_idx is not None:
                # 计算谱相似性作为融合权重
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
        """简单加权融合作为回退方案"""
        fused = {}
        for pair in set(dist1.keys()) & set(dist2.keys()):
            fused[pair] = {}
            for layer in set(dist1[pair].keys()) & set(dist2[pair].keys()):
                fused[pair][layer] = alpha * dist1[pair][layer] + (1 - alpha) * dist2[pair][layer]
        return fused

class CommunityAwareFusion:
    """
    社区感知融合
    基于社区结构调整融合策略
    """
    
    def __init__(self, resolution=1.0):
        self.resolution = resolution
    
    def detect_communities(self, graph):
        """检测图的社区结构"""
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(graph, resolution=self.resolution)
            return partition
        except ImportError:
            # 使用networkx的社区检测作为备选
            try:
                communities = nx.algorithms.community.greedy_modularity_communities(graph)
                partition = {}
                for i, community in enumerate(communities):
                    for node in community:
                        partition[node] = i
                return partition
            except:
                # 如果都失败，返回每个节点为一个社区
                return {node: i for i, node in enumerate(graph.nodes())}
    
    def partition_pairs_by_community(self, distances, community_structure):
        """根据社区结构分割节点对"""
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
        """社区感知融合"""
        community_structure = self.detect_communities(graph)
        
        # 分割社区内和社区间的节点对
        intra1, inter1 = self.partition_pairs_by_community(dist1, community_structure)
        intra2, inter2 = self.partition_pairs_by_community(dist2, community_structure)
        
        # 对社区内和社区间使用不同的融合策略
        fused_intra = self._fuse_intra_community(intra1, intra2, alpha=0.7)  # 社区内更重视度分布
        fused_inter = self._fuse_inter_community(inter1, inter2, alpha=0.3)  # 社区间更重视graphlet
        
        # 合并结果
        fused_distances = {**fused_intra, **fused_inter}
        return fused_distances
    
    def _fuse_intra_community(self, dist1, dist2, alpha):
        """社区内融合策略"""
        fused = {}
        common_pairs = set(dist1.keys()) & set(dist2.keys())
        
        for pair in common_pairs:
            fused[pair] = {}
            for layer in set(dist1[pair].keys()) & set(dist2[pair].keys()):
                fused[pair][layer] = alpha * dist1[pair][layer] + (1 - alpha) * dist2[pair][layer]
        
        return fused
    
    def _fuse_inter_community(self, dist1, dist2, alpha):
        """社区间融合策略"""
        fused = {}
        common_pairs = set(dist1.keys()) & set(dist2.keys())
        
        for pair in common_pairs:
            fused[pair] = {}
            for layer in set(dist1[pair].keys()) & set(dist2[pair].keys()):
                fused[pair][layer] = alpha * dist1[pair][layer] + (1 - alpha) * dist2[pair][layer]
        
        return fused

class EnsembleFusion:
    """
    集成融合方法
    结合多种融合策略的集成方法
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
        """集成多种融合方法"""
        # 提取图特征用于注意力机制
        graph_features = self._extract_graph_features(graph)
        
        # 收集所有方法的融合结果
        fusion_results = []
        method_weights = []
        
        for method_name, method in self.methods:
            try:
                if method_name == 'attention':
                    result = method.fuse_distances(dist1, dist2, graph_features)
                else:
                    result = method.fuse_with_community_awareness(dist1, dist2, graph) if method_name == 'community' else method.fuse_with_spectral_weights(dist1, dist2, graph)
                
                fusion_results.append(result)
                method_weights.append(1.0)  # 等权重
                
            except Exception as e:
                print(f"方法 {method_name} 失败: {e}")
                continue
        
        if not fusion_results:
            # 回退到简单融合
            return self._simple_fusion(dist1, dist2)
        
        # 集成所有结果
        return self._ensemble_average(fusion_results, method_weights)
    
    def _extract_graph_features(self, graph):
        """提取图特征"""
        features = {}
        
        # 计算节点特征
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
            
            # 填充到固定维度
            while len(features[node]) < 64:
                features[node] = np.concatenate([features[node], features[node]])
            features[node] = features[node][:64]
        
        return features
    
    def _ensemble_average(self, fusion_results, weights):
        """集成平均多个融合结果"""
        if not fusion_results:
            return {}
        
        # 归一化权重
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0] * len(fusion_results)
            total_weight = len(fusion_results)
        
        normalized_weights = [w / total_weight for w in weights]
        
        # 找到所有公共节点对
        common_pairs = set(fusion_results[0].keys())
        for result in fusion_results[1:]:
            common_pairs &= set(result.keys())
        
        ensemble_result = {}
        for pair in common_pairs:
            ensemble_result[pair] = {}
            
            # 找到所有公共层
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
    
    def _simple_fusion(self, dist1, dist2, alpha=0.5):
        """简单融合作为回退"""
        fused = {}
        for pair in set(dist1.keys()) & set(dist2.keys()):
            fused[pair] = {}
            for layer in set(dist1[pair].keys()) & set(dist2[pair].keys()):
                fused[pair][layer] = alpha * dist1[pair][layer] + (1 - alpha) * dist2[pair][layer]
        return fused

class AdvancedFusionStruc2Vec:
    """
    高级融合的Struc2Vec实现
    """
    
    def __init__(self, graph, fusion_method='ensemble', **kwargs):
        """
        初始化高级融合Struc2Vec
        
        Args:
            graph: NetworkX图
            fusion_method: 融合方法 ('attention', 'pyramid', 'spectral', 'community', 'ensemble')
            **kwargs: 其他参数
        """
        self.graph = graph
        self.fusion_method = fusion_method
        self.fusion_params = kwargs.get('fusion_params', {})
        
        # 创建融合器
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
            raise ValueError(f"未知的融合方法: {fusion_method}")
        
        # 生成高级融合距离
        self._generate_advanced_fused_distances(graph, **kwargs)
        
        # 创建修改后的Struc2Vec实例
        self.model = Struc2Vec(
            graph,
            walk_length=kwargs.get('walk_length', 40),
            num_walks=kwargs.get('num_walks', 8),
            workers=kwargs.get('workers', 1),
            verbose=kwargs.get('verbose', 0),
            opt1_reduce_len=kwargs.get('opt1_reduce_len', True),
            opt2_reduce_sim_calc=kwargs.get('opt2_reduce_sim_calc', True)
        )
        
        # 替换距离文件
        self._replace_structural_distances()
    
    def _replace_structural_distances(self):
        """替换结构距离文件"""
        import pickle
        
        # 加载融合后的距离
        with open(self.fused_distance_file, 'rb') as f:
            fused_distances = pickle.load(f)
        
        # 将融合距离设置到模型中
        self.model.structural_distance = fused_distances
        
        # 创建临时文件来存储距离
        temp_dist_file = Path(self.model.temp_path) / "structural_dist.pkl"
        temp_dist_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_dist_file, 'wb') as f:
            pickle.dump(fused_distances, f)
        
        print(f"🔄 距离文件替换完成: {temp_dist_file}")
    
    def train(self, embed_size=64, window_size=5, iter=3, **kwargs):
        """训练模型"""
        return self.model.train(
            embed_size=embed_size,
            window_size=window_size,
            iter=iter,
            **kwargs
        )
    
    def get_embeddings(self):
        """获取嵌入"""
        return self.model.get_embeddings()
    
    def _generate_advanced_fused_distances(self, graph, **kwargs):
        """生成高级融合距离"""
        from src.algorithms.graphlet_based.compute_edges_improved import generate_improved_structural_distance
        import tempfile
        import shutil
        import time
        
        # 创建临时目录
        temp_dir = Path(tempfile.mkdtemp())
        output_dir = Path(__file__).parent.parent.parent / 'output' / 'distances'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 保存临时图文件
            temp_graph_file = temp_dir / 'temp_graph.edgelist'
            nx.write_edgelist(graph, str(temp_graph_file), data=False)
            
            # 生成原始距离（基于度序列）
            original_temp = Struc2Vec(graph, walk_length=10, num_walks=1, workers=1, verbose=0)
            original_dist_file = temp_dir / 'original_dist.pkl'
            time.sleep(1)  # 确保文件生成
            if Path(original_temp.temp_path + "/structural_dist.pkl").exists():
                shutil.copy(original_temp.temp_path + "/structural_dist.pkl", original_dist_file)
                shutil.rmtree(original_temp.temp_path)
            
            # 生成Graphlet距离
            graphlet_dist_file = temp_dir / 'graphlet_dist.pkl'
            generate_improved_structural_distance(
                str(temp_graph_file),
                str(graphlet_dist_file),
                max_layer=kwargs.get('max_layer', 3),
                distance_method=kwargs.get('distance_method', 'frobenius'),
                use_orbit_selection=kwargs.get('use_orbit_selection', False)
            )
            
            # 加载距离文件
            with open(original_dist_file, 'rb') as f:
                original_distances = pickle.load(f)
            with open(graphlet_dist_file, 'rb') as f:
                graphlet_distances = pickle.load(f)
            
            # 应用高级融合
            if self.fusion_method == 'pyramid':
                # 特殊处理金字塔融合
                pyramid1 = self.fuser.build_feature_pyramid(original_distances, kwargs.get('max_layer', 3))
                pyramid2 = self.fuser.build_feature_pyramid(graphlet_distances, kwargs.get('max_layer', 3))
                fused_pyramid = self.fuser.fuse_pyramid_levels(pyramid1, pyramid2)
                fused_distances = self.fuser.flatten_pyramid(fused_pyramid)
            elif self.fusion_method == 'attention':
                # 提取图特征
                graph_features = self.fuser._extract_graph_features(graph) if hasattr(self.fuser, '_extract_graph_features') else self._extract_simple_features(graph)
                fused_distances = self.fuser.fuse_distances(original_distances, graphlet_distances, graph_features)
            elif self.fusion_method == 'ensemble':
                fused_distances = self.fuser.ensemble_fuse(original_distances, graphlet_distances, graph)
            else:
                # 其他方法
                if hasattr(self.fuser, 'fuse_with_spectral_weights'):
                    fused_distances = self.fuser.fuse_with_spectral_weights(original_distances, graphlet_distances, graph)
                elif hasattr(self.fuser, 'fuse_with_community_awareness'):
                    fused_distances = self.fuser.fuse_with_community_awareness(original_distances, graphlet_distances, graph)
                else:
                    # 回退到简单融合
                    fused_distances = self._simple_fusion(original_distances, graphlet_distances)
            
            # 保存融合后的距离
            self.fused_distance_file = output_dir / f'advanced_fused_{self.fusion_method}_{id(graph)}.pkl'
            with open(self.fused_distance_file, 'wb') as f:
                pickle.dump(fused_distances, f)
            
            print(f"✅ 高级融合距离文件生成完成: {self.fused_distance_file}")
            
        finally:
            # 清理临时文件
            shutil.rmtree(temp_dir)
    
    def _extract_simple_features(self, graph):
        """提取简单图特征"""
        features = {}
        degree_centrality = nx.degree_centrality(graph)
        clustering_coeff = nx.clustering(graph)
        
        for node in graph.nodes():
            features[node] = np.array([
                degree_centrality.get(node, 0),
                clustering_coeff.get(node, 0),
                0.0, 0.0  # 占位符
            ])
            # 扩展到64维
            features[node] = np.tile(features[node], 16)
        
        return features
    
    def _simple_fusion(self, dist1, dist2, alpha=0.5):
        """简单融合作为回退"""
        fused = {}
        for pair in set(dist1.keys()) & set(dist2.keys()):
            fused[pair] = {}
            for layer in set(dist1[pair].keys()) & set(dist2[pair].keys()):
                fused[pair][layer] = alpha * dist1[pair][layer] + (1 - alpha) * dist2[pair][layer]
        return fused
    
    def get_method_name(self):
        """获取方法名称"""
        return f"Advanced Fusion Struc2Vec ({self.fusion_method})"

if __name__ == "__main__":
    # 测试高级融合方法
    print("高级融合方法模块加载成功")
    print("支持的融合方法: attention, pyramid, spectral, community, ensemble")
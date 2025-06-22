#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级特征融合策略
"""
import os
import sys
import pickle
import numpy as np
import networkx as nx

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def node_based_adaptive_fusion(G, dist1_path, dist2_path, output_path):
    """基于节点特性的自适应融合"""
    with open(dist1_path, 'rb') as f:
        dist1 = pickle.load(f)
    with open(dist2_path, 'rb') as f:
        dist2 = pickle.load(f)
    
    # 计算节点特性
    clustering = nx.clustering(G)
    degrees = dict(G.degree())
    
    # 计算全局特性
    avg_clustering = np.mean(list(clustering.values()))
    avg_degree = np.mean(list(degrees.values()))
    
    fused = {}
    
    for pair in set(dist1.keys()).intersection(dist2.keys()):
        u, v = pair
        
        # 节点特性权重（直接使用字符串节点ID）
        u_clustering = clustering.get(u, 0)
        v_clustering = clustering.get(v, 0) 
        u_degree = degrees.get(u, 0)
        v_degree = degrees.get(v, 0)
        
        # 自适应权重计算
        pair_clustering = (u_clustering + v_clustering) / 2
        pair_degree = (u_degree + v_degree) / 2
        
        # 高聚类系数的节点对更信任 graphlet
        clustering_factor = min(pair_clustering / (avg_clustering + 1e-6), 2.0)
        
        # 高度节点对更信任度序列
        degree_factor = min(pair_degree / (avg_degree + 1e-6), 2.0)
        
        # 动态权重：聚类系数高 -> 更信任graphlet，度高 -> 更信任度序列
        alpha = 0.5 + 0.2 * clustering_factor - 0.1 * degree_factor
        alpha = max(0.1, min(0.9, alpha))  # 限制在[0.1, 0.9]
        
        layers1 = dist1[pair]
        layers2 = dist2[pair]
        fused[pair] = {}
        
        for layer in set(layers1.keys()).intersection(layers2.keys()):
            # 层次自适应：低层更信任度序列，高层可以更信任graphlet
            layer_bias = min(layer * 0.1, 0.3)
            final_alpha = alpha + layer_bias
            final_alpha = max(0.1, min(0.9, final_alpha))
            
            fused[pair][layer] = final_alpha * layers2[layer] + (1 - final_alpha) * layers1[layer]
    
    with open(output_path, 'wb') as f:
        pickle.dump(fused, f)
    
    print(f"✅ 自适应融合完成: {output_path}")

def ensemble_fusion(dist1_path, dist2_path, output_path):
    """集成融合：使用多种策略的组合"""
    with open(dist1_path, 'rb') as f:
        dist1 = pickle.load(f)
    with open(dist2_path, 'rb') as f:
        dist2 = pickle.load(f)
    
    fused = {}
    
    for pair in set(dist1.keys()).intersection(dist2.keys()):
        layers1 = dist1[pair]
        layers2 = dist2[pair]
        fused[pair] = {}
        
        for layer in set(layers1.keys()).intersection(layers2.keys()):
            d1 = layers1[layer]
            d2 = layers2[layer]
            
            # 多种融合策略的加权组合
            strategies = [
                0.5 * d1 + 0.5 * d2,           # 均值
                min(d1, d2),                    # 最小值
                np.sqrt(d1 * d2),              # 几何平均
                2 * d1 * d2 / (d1 + d2 + 1e-8) # 调和平均
            ]
            
            # 权重：均值0.4，最小值0.3，几何平均0.2，调和平均0.1
            weights = [0.4, 0.3, 0.2, 0.1]
            fused[pair][layer] = sum(w * s for w, s in zip(weights, strategies))
    
    with open(output_path, 'wb') as f:
        pickle.dump(fused, f)
    
    print(f"✅ 集成融合完成: {output_path}")

def confidence_based_fusion(dist1_path, dist2_path, output_path):
    """基于置信度的融合"""
    with open(dist1_path, 'rb') as f:
        dist1 = pickle.load(f)
    with open(dist2_path, 'rb') as f:
        dist2 = pickle.load(f)
    
    # 计算每个距离的统计特性
    all_dist1 = []
    all_dist2 = []
    
    for pair in dist1:
        for layer in dist1[pair]:
            all_dist1.append(dist1[pair][layer])
    
    for pair in dist2:
        for layer in dist2[pair]:
            all_dist2.append(dist2[pair][layer])
    
    # 分布特性
    mean1, std1 = np.mean(all_dist1), np.std(all_dist1)
    mean2, std2 = np.mean(all_dist2), np.std(all_dist2)
    
    fused = {}
    
    for pair in set(dist1.keys()).intersection(dist2.keys()):
        layers1 = dist1[pair]
        layers2 = dist2[pair]
        fused[pair] = {}
        
        for layer in set(layers1.keys()).intersection(layers2.keys()):
            d1 = layers1[layer]
            d2 = layers2[layer]
            
            # 计算置信度（距离均值越近，置信度越高）
            conf1 = 1 / (1 + abs(d1 - mean1) / (std1 + 1e-8))
            conf2 = 1 / (1 + abs(d2 - mean2) / (std2 + 1e-8))
            
            # 归一化权重
            total_conf = conf1 + conf2
            alpha = conf2 / total_conf if total_conf > 0 else 0.5
            
            fused[pair][layer] = alpha * d2 + (1 - alpha) * d1
    
    with open(output_path, 'wb') as f:
        pickle.dump(fused, f)
    
    print(f"✅ 置信度融合完成: {output_path}")

def main():
    """测试高级融合策略"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "output")
    
    # 距离文件路径
    original_dist = os.path.join(output_dir, "structural_dist_original.pkl")
    graphlet_dist = os.path.join(output_dir, "structural_dist_brazil-airports.pkl")
    
    if not os.path.exists(original_dist):
        print("❌ 需要先运行 simple_fusion_evaluation.py 生成原始距离文件")
        return
    
    if not os.path.exists(graphlet_dist):
        print("❌ 需要先运行 generate_simple_distance.py 生成 graphlet 距离文件")
        return
    
    # 加载图 - 使用字符串节点类型以与距离文件匹配
    graph_path = os.path.join(base_dir, "data/flight/brazil-airports.edgelist")
    G = nx.read_edgelist(graph_path, nodetype=str, create_using=nx.DiGraph())
    
    print("=" * 60)
    print("生成高级融合距离文件")
    print("=" * 60)
    
    # 1. 自适应融合
    adaptive_path = os.path.join(output_dir, "fused_adaptive.pkl")
    node_based_adaptive_fusion(G, original_dist, graphlet_dist, adaptive_path)
    
    # 2. 集成融合
    ensemble_path = os.path.join(output_dir, "fused_ensemble.pkl")
    ensemble_fusion(original_dist, graphlet_dist, ensemble_path)
    
    # 3. 置信度融合
    confidence_path = os.path.join(output_dir, "fused_confidence.pkl")
    confidence_based_fusion(original_dist, graphlet_dist, confidence_path)
    
    print(f"\n✅ 高级融合文件已生成！")
    print(f"\n现在可以运行以下命令测试：")
    print(f"python simple_evaluation.py  # 基础对比")
    print(f"python test_advanced_fusion.py  # 测试高级融合方法")

if __name__ == "__main__":
    main()
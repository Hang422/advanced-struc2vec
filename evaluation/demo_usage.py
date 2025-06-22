#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示如何使用改进的 graphlet-enhanced struc2vec
"""
import os
import sys
import time
import numpy as np
import networkx as nx

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from algorithms.traditional.struc2vec import Struc2Vec

def demo_improved_struc2vec():
    """演示使用改进的 graphlet 距离的 struc2vec"""
    
    print("🚀 演示改进的 Graphlet-Enhanced Struc2vec")
    print("=" * 60)
    
    # 准备路径
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    graph_path = os.path.join(project_root, "data/flight/brazil-airports.edgelist")
    output_dir = os.path.join(project_root, "output/")
    
    # 1. 加载图
    print("1. 加载巴西机场网络...")
    G = nx.read_edgelist(graph_path, create_using=nx.DiGraph(), 
                        nodetype=None, data=[('weight', int)])
    print(f"   图信息: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    
    # 2. 使用不同的距离文件进行比较
    distance_files = {
        "改进版(完整)": os.path.join(output_dir, "structural_dist_improved_basic_brazil-airports.pkl"),
        "改进版(精简)": os.path.join(output_dir, "structural_dist_improved_compact_brazil-airports.pkl"),
        "改进版(Frobenius)": os.path.join(output_dir, "structural_dist_improved_frobenius_brazil-airports.pkl")
    }
    
    results = {}
    
    for name, dist_file in distance_files.items():
        if not os.path.exists(dist_file):
            print(f"⚠️ 跳过 {name}: 距离文件不存在")
            print(f"   请先运行: python simple_evaluation.py brazil-airports")
            continue
            
        print(f"\n2. 训练 {name}...")
        start_time = time.time()
        
        # 创建模型
        model = Struc2Vec(
            G, 
            num_walks=10,          # 减少随机游走数量加速演示
            walk_length=80, 
            workers=2,             # 减少工作进程
            verbose=0,             # 减少输出
            opt1_reduce_len=True, 
            opt2_reduce_sim_calc=True,
            structural_dist_file=dist_file
        )
        
        # 训练模型
        model.train(embed_size=64, window_size=5, workers=2, iter=3)  # 减少维度和迭代次数
        embeddings = model.get_embeddings()
        
        training_time = time.time() - start_time
        
        # 计算嵌入质量指标
        embedding_quality = evaluate_embedding_quality(embeddings, G)
        
        results[name] = {
            'time': training_time,
            'embedding_size': len(embeddings),
            'quality': embedding_quality
        }
        
        print(f"   ✅ 完成: {training_time:.2f}秒, 嵌入维度: {list(embeddings.values())[0].shape}")
    
    # 3. 结果比较
    print(f"\n{'='*60}")
    print("结果比较:")
    print(f"{'='*60}")
    
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  训练时间: {result['time']:.2f} 秒")
        print(f"  嵌入节点数: {result['embedding_size']}")
        for metric, value in result['quality'].items():
            print(f"  {metric}: {value:.4f}")
    
    # 4. 演示嵌入的使用
    if results:
        print(f"\n{'='*60}")
        print("嵌入使用示例:")
        print(f"{'='*60}")
        
        # 使用第一个可用的嵌入
        first_method = list(results.keys())[0]
        dist_file = distance_files[first_method]
        
        model = Struc2Vec(G, 5, 40, workers=1, verbose=0,
                         structural_dist_file=dist_file)
        model.train(embed_size=32, iter=1)
        embeddings = model.get_embeddings()
        
        # 找到最相似的节点对
        print("3. 寻找最相似的机场...")
        similarities = compute_node_similarities(embeddings)
        
        print("   前5个最相似的机场对:")
        for i, (node1, node2, sim) in enumerate(similarities[:5]):
            print(f"   {i+1}. 机场 {node1} - 机场 {node2}: 相似度 {sim:.4f}")


def evaluate_embedding_quality(embeddings, graph):
    """简单的嵌入质量评估"""
    nodes = list(embeddings.keys())
    vectors = np.array([embeddings[node] for node in nodes])
    
    # 计算向量的基本统计量
    mean_norm = np.mean([np.linalg.norm(v) for v in vectors])
    std_norm = np.std([np.linalg.norm(v) for v in vectors])
    
    # 计算向量间的平均余弦相似度
    from scipy.spatial.distance import pdist, squareform
    try:
        cosine_distances = pdist(vectors, metric='cosine')
        avg_cosine_sim = 1 - np.mean(cosine_distances)
    except:
        avg_cosine_sim = 0.0
    
    return {
        '平均向量范数': mean_norm,
        '范数标准差': std_norm,
        '平均余弦相似度': avg_cosine_sim
    }


def compute_node_similarities(embeddings, top_k=10):
    """计算节点间的相似度"""
    from scipy.spatial.distance import cosine
    
    nodes = list(embeddings.keys())
    similarities = []
    
    for i in range(len(nodes)):
        for j in range(i+1, min(i+50, len(nodes))):  # 只计算部分以节省时间
            node1, node2 = nodes[i], nodes[j]
            try:
                sim = 1 - cosine(embeddings[node1], embeddings[node2])
                similarities.append((node1, node2, sim))
            except:
                continue
    
    # 按相似度排序
    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities[:top_k]


if __name__ == "__main__":
    try:
        demo_improved_struc2vec()
        print(f"\n🎉 演示完成!")
        print(f"\n💡 提示:")
        print(f"   - 精简版适合快速实验和大图")
        print(f"   - 完整版提供更丰富的特征")
        print(f"   - Frobenius版计算最快")
        print(f"   - 可以根据具体任务选择合适的版本")
        
    except Exception as e:
        print(f"❌ 演示出错: {e}")
        print(f"\n🔧 解决方案:")
        print(f"   1. 确保已生成距离文件: python simple_evaluation.py brazil-airports")
        print(f"   2. 检查依赖是否安装: scipy, numpy, networkx")
        import traceback
        traceback.print_exc()
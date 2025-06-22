#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成简单的 graphlet 结构距离文件用于测试
"""
import os
import sys

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algorithms.graphlet_based.compute_edges_improved import generate_improved_structural_distance

def main():
    """生成 brazil-airports 的结构距离文件"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    graph_path = os.path.join(base_dir, 'data/flight/brazil-airports.edgelist')
    output_path = os.path.join(base_dir, 'output/structural_dist_brazil-airports.pkl')
    
    print("生成 Brazil Airports 的 graphlet 距离文件...")
    print("=" * 60)
    
    try:
        generate_improved_structural_distance(
            graph_path,
            output_path,
            max_layer=3,  # 减少层数以加快计算
            distance_method='frobenius',  # 使用更快的距离度量
            use_orbit_selection=False,  # 不进行 orbit 选择
            top_k_orbits=40
        )
        print(f"\n✅ 成功生成: {output_path}")
        
    except Exception as e:
        print(f"\n❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
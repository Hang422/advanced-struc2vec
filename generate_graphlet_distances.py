#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 graphlet 结构距离文件
"""
import os
import sys
import time

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algorithms.graphlet_based.compute_edges_improved import (
    generate_improved_structural_distance,
    compute_node_gdv,
    preprocess_edgelist
)

def generate_all_distances():
    """生成所有需要的结构距离文件"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    datasets = {
        'brazil-airports': os.path.join(base_dir, 'data/flight/brazil-airports.edgelist'),
        'wiki': os.path.join(base_dir, 'data/wiki/Wiki_edgelist.txt'),
        'lastfm': os.path.join(base_dir, 'data/lastfm_asia/lastfm_asia.edgelist')
    }
    
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    methods = {
        'basic': {'distance_method': 'euclidean', 'gdv_method': 'basic'},
        'compact': {'distance_method': 'euclidean', 'gdv_method': 'compact'},
        'frobenius': {'distance_method': 'frobenius', 'gdv_method': 'basic'}
    }
    
    for dataset_name, graph_path in datasets.items():
        if not os.path.exists(graph_path):
            print(f"跳过 {dataset_name}: 文件不存在")
            continue
            
        print(f"\n处理数据集: {dataset_name}")
        print("=" * 60)
        
        # 1. 生成原始 graphlet 距离
        print(f"生成基础 graphlet 距离...")
        try:
            # 生成基础距离文件
            output_path = os.path.join(output_dir, f'structural_dist_{dataset_name}.pkl')
            generate_improved_structural_distance(
                graph_path,
                output_path,
                max_layer=5,
                distance_method='euclidean',
                use_orbit_selection=False
            )
            print(f"  ✅ 基础距离文件: {output_path}")
            
        except Exception as e:
            print(f"  ❌ 生成基础距离失败: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # 2. 生成改进版本
        for method_name, params in methods.items():
            print(f"\n生成改进版本 ({method_name})...")
            output_path = os.path.join(output_dir, f'structural_dist_improved_{method_name}_{dataset_name}.pkl')
            
            try:
                start_time = time.time()
                generate_improved_structural_distance(
                    graph_path,
                    output_path,
                    max_layer=5,
                    distance_method=params['distance_method'],
                    gdv_method=params['gdv_method'],
                    use_orbit_selection=True,
                    top_k_orbits=40
                )
                elapsed = time.time() - start_time
                print(f"  ✅ {method_name}: {output_path} (耗时: {elapsed:.2f}秒)")
                
            except Exception as e:
                print(f"  ❌ {method_name} 失败: {e}")
    
    print(f"\n所有距离文件生成完成!")
    print(f"输出目录: {output_dir}")
    
    # 列出生成的文件
    print(f"\n生成的文件:")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith('.pkl'):
            size = os.path.getsize(os.path.join(output_dir, f)) / 1024 / 1024
            print(f"  {f} ({size:.2f} MB)")

if __name__ == "__main__":
    generate_all_distances()
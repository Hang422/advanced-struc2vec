#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载工具
"""
import sys
import os
import networkx as nx
from pathlib import Path
from typing import Tuple, Dict, Any

# Add parent project path
parent_path = Path(__file__).parent.parent.parent.parent
sys.path.append(str(parent_path))

from GraphEmbedding.ge.classify import read_node_label

class DataLoader:
    """Data loader"""
    
    def __init__(self):
        """Initializing数据加载器"""
        from ..config.config import DATA_CONFIG, DATASETS, PARENT_ROOT
        self.data_config = DATA_CONFIG
        self.datasets = DATASETS
        self.parent_root = PARENT_ROOT
    
    def load_dataset(self, dataset_name: str) -> Tuple[nx.Graph, Tuple[list, list]]:
        """
        Loading dataset
        
        Args:
            dataset_name: dataset名称
            
        Returns:
            (graph, (X, Y)): 图对象和标签数据
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"未知dataset: {dataset_name}. 可用dataset: {list(self.datasets.keys())}")
        
        dataset_info = self.datasets[dataset_name]
        
        # 构建文件路径 - 使用原始数据路径
        original_data_path = self.parent_root / 'data'
        graph_path = original_data_path / dataset_info['graph_file']
        label_path = original_data_path / dataset_info['label_file']
        
        print(f"📂 Loading dataset: {dataset_name}")
        print(f"   描述: {dataset_info['description']}")
        print(f"   Graph file: {graph_path}")
        print(f"   Label file: {label_path}")
        
        # Load graph
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph file不存在: {graph_path}")
        
        try:
            # 尝试不同的加载方式
            G = nx.read_edgelist(str(graph_path), nodetype=str, create_using=nx.DiGraph())
            print(f"   ✅ 图加载Success: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        except Exception as e:
            print(f"   ❌ 图加载Failed: {e}")
            raise
        
        # Load labels
        if not label_path.exists():
            raise FileNotFoundError(f"Label file不存在: {label_path}")
        
        try:
            X, Y = read_node_label(str(label_path), skip_head=True)
            print(f"   ✅ 标签加载Success: {len(X)} 个标记nodes")
        except Exception as e:
            print(f"   ❌ 标签加载Failed: {e}")
            raise
        
        return G, (X, Y)
    
    def save_temp_graph(self, graph: nx.Graph) -> Path:
        """
        Save temporary graph file
        
        Args:
            graph: NetworkX graph对象
            
        Returns:
            临时文件路径
        """
        temp_dir = self.data_config['temp_dir']
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        temp_file = temp_dir / f"temp_graph_{id(graph)}.edgelist"
        
        # Saving图
        nx.write_edgelist(graph, str(temp_file), data=False)
        
        print(f"💾 临时Graph fileSaving: {temp_file}")
        return temp_file
    
    def copy_external_data(self, dataset_name: str) -> None:
        """
        复制外部数据到项目目录
        
        Args:
            dataset_name: dataset名称
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"未知dataset: {dataset_name}")
        
        dataset_info = self.datasets[dataset_name]
        
        # 源路径
        original_data_path = self.parent_root / 'data'
        source_graph = original_data_path / dataset_info['graph_file']
        source_label = original_data_path / dataset_info['label_file']
        
        # 目标路径
        target_dir = self.data_config['raw_data_dir'] / dataset_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        import shutil
        
        # 复制文件
        if source_graph.exists():
            target_graph = target_dir / source_graph.name
            shutil.copy2(source_graph, target_graph)
            print(f"📋 复制Graph file: {source_graph} -> {target_graph}")
        
        if source_label.exists():
            target_label = target_dir / source_label.name
            shutil.copy2(source_label, target_label)
            print(f"📋 复制Label file: {source_label} -> {target_label}")
    
    def get_dataset_info(self, dataset_name: str = None) -> Dict[str, Any]:
        """
        获取dataset信息
        
        Args:
            dataset_name: dataset名称，None表示获取所有dataset信息
            
        Returns:
            dataset信息字典
        """
        if dataset_name is None:
            return self.datasets.copy()
        
        if dataset_name not in self.datasets:
            raise ValueError(f"未知dataset: {dataset_name}")
        
        return self.datasets[dataset_name].copy()
    
    def list_available_datasets(self) -> list:
        """
        列出可用的dataset
        
        Returns:
            dataset名称列表
        """
        return list(self.datasets.keys())

if __name__ == "__main__":
    # 测试数据加载器
    loader = DataLoader()
    
    print("可用dataset:")
    for name in loader.list_available_datasets():
        info = loader.get_dataset_info(name)
        print(f"  - {name}: {info['description']}")
    
    # 测试Loading dataset
    try:
        G, (X, Y) = loader.load_dataset('brazil-airports')
        print(f"\n测试加载Success!")
    except Exception as e:
        print(f"\n测试加载Failed: {e}")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graphlet 增强 Struc2Vec 算法实现
"""
import sys
import os
import pickle
from pathlib import Path

# 添加父项目路径
parent_path = Path(__file__).parent.parent.parent.parent
sys.path.append(str(parent_path))

from .base import BaseStruc2Vec
from libs.GraphEmbedding.ge.models.struc2vec import Struc2Vec
from src.algorithms.graphlet_based.compute_edges_improved import generate_improved_structural_distance

class GraphletStruc2Vec(BaseStruc2Vec):
    """Graphlet 增强 Struc2Vec 算法"""
    
    def __init__(self, graph, **kwargs):
        super().__init__(graph, **kwargs)
        self.model = None
        self.distance_file = None
        
        # 默认参数
        self.walk_length = kwargs.get('walk_length', 40)
        self.num_walks = kwargs.get('num_walks', 10)
        self.embed_size = kwargs.get('embed_size', 64)
        self.window_size = kwargs.get('window_size', 5)
        self.workers = kwargs.get('workers', 2)
        self.iter = kwargs.get('iter', 3)
        self.verbose = kwargs.get('verbose', 0)
        
        # Graphlet 特定参数
        self.max_layer = kwargs.get('max_layer', 3)
        self.k = kwargs.get('k', 5)
        self.distance_method = kwargs.get('distance_method', 'frobenius')
        self.use_orbit_selection = kwargs.get('use_orbit_selection', False)
        self.top_k_orbits = kwargs.get('top_k_orbits', 40)
        
        # 文件路径
        self.temp_graph_path = kwargs.get('temp_graph_path')
        self.distance_output_path = kwargs.get('distance_output_path')
    
    def _prepare_graph_file(self):
        """准备图文件"""
        if self.temp_graph_path is None:
            # 创建临时图文件
            from ..utils.data_loader import DataLoader
            loader = DataLoader()
            self.temp_graph_path = loader.save_temp_graph(self.graph)
        
        return self.temp_graph_path
    
    def _generate_distance_file(self):
        """生成 Graphlet 距离文件"""
        print("📊 生成 Graphlet 结构距离...")
        
        graph_file = self._prepare_graph_file()
        
        if self.distance_output_path is None:
            # 使用默认输出路径
            from ..config.config import DATA_CONFIG
            output_dir = DATA_CONFIG['distances_dir']
            output_dir.mkdir(parents=True, exist_ok=True)
            self.distance_output_path = output_dir / f"graphlet_distances_{id(self.graph)}.pkl"
        
        # 生成距离文件
        generate_improved_structural_distance(
            str(graph_file),
            str(self.distance_output_path),
            k=self.k,
            max_layer=self.max_layer,
            distance_method=self.distance_method,
            use_orbit_selection=self.use_orbit_selection,
            top_k_orbits=self.top_k_orbits
        )
        
        return str(self.distance_output_path)
    
    def train(self):
        """训练 Graphlet 增强 Struc2Vec 模型"""
        print("🚀 训练 Graphlet 增强 Struc2Vec...")
        
        # 生成距离文件
        self.distance_file = self._generate_distance_file()
        
        # 创建模型
        self.model = Struc2Vec(
            self.graph,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            workers=self.workers,
            verbose=self.verbose,
            opt1_reduce_len=True,
            opt2_reduce_sim_calc=True,
            structural_dist_file=self.distance_file
        )
        
        # 训练
        self.model.train(
            embed_size=self.embed_size,
            window_size=self.window_size,
            workers=self.workers,
            iter=self.iter
        )
        
        # 获取嵌入
        self.embeddings = self.model.get_embeddings()
        
        print("✅ Graphlet 增强 Struc2Vec 训练完成")
    
    def get_embeddings(self):
        """获取节点嵌入"""
        if self.embeddings is None:
            raise ValueError("需要先训练模型")
        return self.embeddings
    
    def get_method_name(self):
        """获取方法名称"""
        return f"Graphlet Struc2Vec (k={self.k}, {self.distance_method})"
    
    def get_distance_file(self):
        """获取距离文件路径"""
        return self.distance_file
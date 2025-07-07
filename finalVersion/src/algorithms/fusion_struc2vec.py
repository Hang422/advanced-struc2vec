#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
融合 Struc2Vec 算法实现
"""
import sys
import pickle
from pathlib import Path

# 添加父项目路径
parent_path = Path(__file__).parent.parent.parent.parent
sys.path.append(str(parent_path))

from .base import BaseStruc2Vec
from .original_struc2vec import OriginalStruc2Vec
from .graphlet_struc2vec import GraphletStruc2Vec
from libs.GraphEmbedding.ge.models.struc2vec import Struc2Vec

class FusionStruc2Vec(BaseStruc2Vec):
    """融合版本 Struc2Vec 算法"""
    
    def __init__(self, graph, **kwargs):
        super().__init__(graph, **kwargs)
        self.model = None
        self.fused_distance_file = None
        
        # 默认参数
        self.walk_length = kwargs.get('walk_length', 40)
        self.num_walks = kwargs.get('num_walks', 10)
        self.embed_size = kwargs.get('embed_size', 64)
        self.window_size = kwargs.get('window_size', 5)
        self.workers = kwargs.get('workers', 2)
        self.iter = kwargs.get('iter', 3)
        self.verbose = kwargs.get('verbose', 0)
        
        # 融合参数
        self.alpha = kwargs.get('alpha', 0.5)  # 融合权重
        self.fusion_method = kwargs.get('fusion_method', 'weighted')  # 融合方法
        
        # 距离文件路径
        self.original_distance_file = kwargs.get('original_distance_file')
        self.graphlet_distance_file = kwargs.get('graphlet_distance_file')
        self.output_distance_file = kwargs.get('output_distance_file')
    
    def _get_or_generate_distance_files(self):
        """获取或生成距离文件"""
        from ..utils.data_loader import DataLoader
        from ..config.config import DATA_CONFIG
        
        loader = DataLoader()
        distances_dir = DATA_CONFIG['distances_dir']
        distances_dir.mkdir(parents=True, exist_ok=True)
        
        # 原始距离文件
        if self.original_distance_file is None:
            print("📊 生成原始结构距离...")
            original_model = OriginalStruc2Vec(self.graph, workers=1, num_walks=1, walk_length=10)
            original_model.model = Struc2Vec(
                self.graph, 
                walk_length=10, 
                num_walks=1, 
                workers=1, 
                verbose=0
            )
            
            # 获取临时距离文件
            import shutil
            import time
            time.sleep(1)  # 等待文件生成
            temp_dist = original_model.model.temp_path + "/structural_dist.pkl"
            
            self.original_distance_file = distances_dir / f"original_distances_{id(self.graph)}.pkl"
            if Path(temp_dist).exists():
                shutil.copy(temp_dist, self.original_distance_file)
                shutil.rmtree(original_model.model.temp_path)
            else:
                raise FileNotFoundError("无法生成原始距离文件")
        
        # Graphlet 距离文件
        if self.graphlet_distance_file is None:
            print("📊 生成 Graphlet 结构距离...")
            graphlet_model = GraphletStruc2Vec(
                self.graph,
                distance_output_path=distances_dir / f"graphlet_distances_{id(self.graph)}.pkl"
            )
            self.graphlet_distance_file = graphlet_model._generate_distance_file()
        
        return str(self.original_distance_file), str(self.graphlet_distance_file)
    
    def _fuse_distances(self, dist1_path, dist2_path, output_path):
        """融合两个距离文件"""
        print(f"🔗 融合距离文件 (方法: {self.fusion_method}, α: {self.alpha})...")
        
        with open(dist1_path, 'rb') as f:
            dist1 = pickle.load(f)
        with open(dist2_path, 'rb') as f:
            dist2 = pickle.load(f)
        
        fused = {}
        
        if self.fusion_method == 'weighted':
            # 线性加权融合
            for pair in set(dist1.keys()).intersection(dist2.keys()):
                fused[pair] = {}
                layers1 = dist1[pair]
                layers2 = dist2[pair]
                for layer in set(layers1.keys()).intersection(layers2.keys()):
                    fused[pair][layer] = self.alpha * layers1[layer] + (1 - self.alpha) * layers2[layer]
        
        elif self.fusion_method == 'min':
            # 取最小值
            for pair in set(dist1.keys()).intersection(dist2.keys()):
                fused[pair] = {}
                layers1 = dist1[pair]
                layers2 = dist2[pair]
                for layer in set(layers1.keys()).intersection(layers2.keys()):
                    fused[pair][layer] = min(layers1[layer], layers2[layer])
        
        elif self.fusion_method == 'max':
            # 取最大值
            for pair in set(dist1.keys()).intersection(dist2.keys()):
                fused[pair] = {}
                layers1 = dist1[pair]
                layers2 = dist2[pair]
                for layer in set(layers1.keys()).intersection(layers2.keys()):
                    fused[pair][layer] = max(layers1[layer], layers2[layer])
        
        # 保存融合结果
        with open(output_path, 'wb') as f:
            pickle.dump(fused, f)
        
        print(f"✅ 融合完成: {output_path}")
        return output_path
    
    def train(self):
        """训练融合 Struc2Vec 模型"""
        print("🚀 训练融合 Struc2Vec...")
        
        # 获取距离文件
        original_dist, graphlet_dist = self._get_or_generate_distance_files()
        
        # 设置输出路径
        if self.output_distance_file is None:
            from ..config.config import DATA_CONFIG
            distances_dir = DATA_CONFIG['distances_dir']
            self.output_distance_file = distances_dir / f"fused_distances_{self.fusion_method}_{self.alpha}_{id(self.graph)}.pkl"
        
        # 融合距离
        self.fused_distance_file = self._fuse_distances(
            original_dist, 
            graphlet_dist, 
            str(self.output_distance_file)
        )
        
        # 创建模型
        self.model = Struc2Vec(
            self.graph,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            workers=self.workers,
            verbose=self.verbose,
            opt1_reduce_len=True,
            opt2_reduce_sim_calc=True,
            structural_dist_file=self.fused_distance_file
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
        
        print("✅ 融合 Struc2Vec 训练完成")
    
    def get_embeddings(self):
        """获取节点嵌入"""
        if self.embeddings is None:
            raise ValueError("需要先训练模型")
        return self.embeddings
    
    def get_method_name(self):
        """获取方法名称"""
        return f"Fusion Struc2Vec ({self.fusion_method}, α={self.alpha})"
    
    def get_fused_distance_file(self):
        """获取融合距离文件路径"""
        return self.fused_distance_file
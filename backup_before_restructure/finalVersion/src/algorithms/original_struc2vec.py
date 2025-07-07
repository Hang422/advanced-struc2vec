#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
原始 Struc2Vec 算法实现
"""
import sys
import os
from pathlib import Path

# 添加父项目路径
parent_path = Path(__file__).parent.parent.parent.parent
sys.path.append(str(parent_path))

from .base import BaseStruc2Vec
from algorithms.traditional.struc2vec import Struc2Vec

class OriginalStruc2Vec(BaseStruc2Vec):
    """原始 Struc2Vec 算法包装类"""
    
    def __init__(self, graph, **kwargs):
        super().__init__(graph, **kwargs)
        self.model = None
        
        # 默认参数
        self.walk_length = kwargs.get('walk_length', 40)
        self.num_walks = kwargs.get('num_walks', 10)
        self.embed_size = kwargs.get('embed_size', 64)
        self.window_size = kwargs.get('window_size', 5)
        self.workers = kwargs.get('workers', 2)
        self.iter = kwargs.get('iter', 3)
        self.opt1_reduce_len = kwargs.get('opt1_reduce_len', True)
        self.opt2_reduce_sim_calc = kwargs.get('opt2_reduce_sim_calc', True)
        self.verbose = kwargs.get('verbose', 0)
    
    def train(self):
        """训练原始 Struc2Vec 模型"""
        print("🚀 训练原始 Struc2Vec...")
        
        # 创建模型
        self.model = Struc2Vec(
            self.graph,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            workers=self.workers,
            verbose=self.verbose,
            opt1_reduce_len=self.opt1_reduce_len,
            opt2_reduce_sim_calc=self.opt2_reduce_sim_calc
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
        
        print("✅ 原始 Struc2Vec 训练完成")
    
    def get_embeddings(self):
        """获取节点嵌入"""
        if self.embeddings is None:
            raise ValueError("需要先训练模型")
        return self.embeddings
    
    def get_method_name(self):
        """获取方法名称"""
        return "Original Struc2Vec"
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础算法接口
"""
from abc import ABC, abstractmethod
import networkx as nx
import numpy as np
from typing import Dict, Any, Optional

class BaseStruc2Vec(ABC):
    """Struc2Vec算法的基础接口"""
    
    def __init__(self, graph: nx.Graph, **kwargs):
        """
        初始化算法
        
        Args:
            graph: NetworkX图对象
            **kwargs: 其他参数
        """
        self.graph = graph
        self.embeddings = None
        self.config = kwargs
        
    @abstractmethod
    def train(self) -> None:
        """训练算法"""
        pass
    
    @abstractmethod
    def get_embeddings(self) -> Dict[str, np.ndarray]:
        """获取节点嵌入"""
        pass
    
    def save_embeddings(self, filepath: str) -> None:
        """保存嵌入到文件"""
        if self.embeddings is None:
            raise ValueError("需要先训练模型")
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.embeddings, f)
    
    def load_embeddings(self, filepath: str) -> Dict[str, np.ndarray]:
        """从文件加载嵌入"""
        import pickle
        with open(filepath, 'rb') as f:
            self.embeddings = pickle.load(f)
        return self.embeddings
    
    def get_config(self) -> Dict[str, Any]:
        """获取算法配置"""
        return self.config.copy()
    
    def set_config(self, **kwargs) -> None:
        """设置算法配置"""
        self.config.update(kwargs)
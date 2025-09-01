#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础algorithm接口
"""
from abc import ABC, abstractmethod
import networkx as nx
import numpy as np
from typing import Dict, Any, Optional

class BaseStruc2Vec(ABC):
    """Struc2Vecalgorithm的基础接口"""
    
    def __init__(self, graph: nx.Graph, **kwargs):
        """
        Initializingalgorithm
        
        Args:
            graph: NetworkX graph对象
            **kwargs: other parameters
        """
        self.graph = graph
        self.embeddings = None
        self.config = kwargs
        
    @abstractmethod
    def train(self) -> None:
        """Trainingalgorithm"""
        pass
    
    @abstractmethod
    def get_embeddings(self) -> Dict[str, np.ndarray]:
        """获取nodes嵌入"""
        pass
    
    def save_embeddings(self, filepath: str) -> None:
        """Saving嵌入到文件"""
        if self.embeddings is None:
            raise ValueError("需要先Trainingmodel")
        
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
        """获取algorithm配置"""
        return self.config.copy()
    
    def set_config(self, **kwargs) -> None:
        """设置algorithm配置"""
        self.config.update(kwargs)
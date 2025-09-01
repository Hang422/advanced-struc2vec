#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Original Struc2Vec algorithm implementation
"""
import sys
import os
from pathlib import Path

# Add parent project path
parent_path = Path(__file__).parent.parent.parent.parent
sys.path.append(str(parent_path))

from .base import BaseStruc2Vec
from libs.GraphEmbedding.ge.models.struc2vec import Struc2Vec

class OriginalStruc2Vec(BaseStruc2Vec):
    """Original Struc2Vec algorithm wrapper class"""
    
    def __init__(self, graph, **kwargs):
        super().__init__(graph, **kwargs)
        self.model = None
        
        # Default parameters
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
        """Train original Struc2Vec model"""
        print("🚀 Training original Struc2Vec...")
        
        # Create model
        self.model = Struc2Vec(
            self.graph,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            workers=self.workers,
            verbose=self.verbose,
            opt1_reduce_len=self.opt1_reduce_len,
            opt2_reduce_sim_calc=self.opt2_reduce_sim_calc
        )
        
        # Train model
        self.model.train(
            embed_size=self.embed_size,
            window_size=self.window_size,
            workers=self.workers,
            iter=self.iter
        )
        
        # Get embeddings
        self.embeddings = self.model.get_embeddings()
        
        print("✅ Original Struc2Vec training completed")
    
    def get_embeddings(self):
        """Get node embeddings"""
        if self.embeddings is None:
            raise ValueError("Model must be trained first")
        return self.embeddings
    
    def get_method_name(self):
        """Get method name"""
        return "Original Struc2Vec"
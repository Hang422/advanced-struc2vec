#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graphlet 增强 Struc2Vec algorithm实现
"""
import sys
import os
import pickle
from pathlib import Path

# Add parent project path
parent_path = Path(__file__).parent.parent.parent.parent
sys.path.append(str(parent_path))

from .base import BaseStruc2Vec
from libs.GraphEmbedding.ge.models.struc2vec import Struc2Vec
from src.algorithms.graphlet_based.compute_edges_improved import generate_improved_structural_distance

class GraphletStruc2Vec(BaseStruc2Vec):
    """Graphlet-enhanced Struc2Vec algorithm"""
    
    def __init__(self, graph, **kwargs):
        super().__init__(graph, **kwargs)
        self.model = None
        self.distance_file = None
        
        # Default parameters
        self.walk_length = kwargs.get('walk_length', 40)
        self.num_walks = kwargs.get('num_walks', 10)
        self.embed_size = kwargs.get('embed_size', 64)
        self.window_size = kwargs.get('window_size', 5)
        self.workers = kwargs.get('workers', 2)
        self.iter = kwargs.get('iter', 3)
        self.verbose = kwargs.get('verbose', 0)
        
        # Graphlet-specific parameters
        self.max_layer = kwargs.get('max_layer', 3)
        self.k = kwargs.get('k', 5)
        self.distance_method = kwargs.get('distance_method', 'frobenius')
        self.use_orbit_selection = kwargs.get('use_orbit_selection', False)
        self.top_k_orbits = kwargs.get('top_k_orbits', 40)
        
        # File paths
        self.temp_graph_path = kwargs.get('temp_graph_path')
        self.distance_output_path = kwargs.get('distance_output_path')
    
    def _prepare_graph_file(self):
        """Prepare graph file"""
        if self.temp_graph_path is None:
            # Create temporary graph file
            from ..utils.data_loader import DataLoader
            loader = DataLoader()
            self.temp_graph_path = loader.save_temp_graph(self.graph)
        
        return self.temp_graph_path
    
    def _generate_distance_file(self):
        """Generate Graphlet distance file"""
        print("📊 Generating Graphlet structural distances...")
        
        graph_file = self._prepare_graph_file()
        
        if self.distance_output_path is None:
            # Use default output path
            from ..config.config import DATA_CONFIG
            output_dir = DATA_CONFIG['distances_dir']
            output_dir.mkdir(parents=True, exist_ok=True)
            self.distance_output_path = output_dir / f"graphlet_distances_{id(self.graph)}.pkl"
        
        # Generate distance file
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
        """Train Graphlet-enhanced Struc2Vec model"""
        print("🚀 Training Graphlet-enhanced Struc2Vec...")
        
        # Generate distance file
        self.distance_file = self._generate_distance_file()
        
        # Create model
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
        
        # Train model
        self.model.train(
            embed_size=self.embed_size,
            window_size=self.window_size,
            workers=self.workers,
            iter=self.iter
        )
        
        # Get embeddings
        self.embeddings = self.model.get_embeddings()
        
        print("✅ Graphlet-enhanced Struc2Vec training completed")
    
    def get_embeddings(self):
        """Get node embeddings"""
        if self.embeddings is None:
            raise ValueError("Model must be trained first")
        return self.embeddings
    
    def get_method_name(self):
        """Get method name"""
        return f"Graphlet Struc2Vec (k={self.k}, {self.distance_method})"
    
    def get_distance_file(self):
        """Get distance file path"""
        return self.distance_file
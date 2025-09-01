#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global configuration file
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
PARENT_ROOT = PROJECT_ROOT.parent

# Data path configuration
DATA_CONFIG = {
    'raw_data_dir': PROJECT_ROOT / 'data' / 'raw',
    'processed_data_dir': PROJECT_ROOT / 'data' / 'processed',
    'temp_dir': PROJECT_ROOT / 'temp',
    'output_dir': PROJECT_ROOT / 'output',
    'distances_dir': PROJECT_ROOT / 'output' / 'distances',
    'embeddings_dir': PROJECT_ROOT / 'output' / 'embeddings',
    'results_dir': PROJECT_ROOT / 'output' / 'results'
}

# Dataset configuration
DATASETS = {
    'brazil-airports': {
        'graph_file': 'flight/brazil-airports.edgelist',
        'label_file': 'flight/labels-brazil-airports.txt',
        'description': 'Brazilian airports network (131 nodes, 1074 edges)'
    },
    'wiki': {
        'graph_file': 'wiki/Wiki_edgelist.txt',
        'label_file': 'wiki/wiki_labels.txt',
        'description': 'Wikipedia network (2405 nodes, 11596 edges)'
    },
    'lastfm': {
        'graph_file': 'lastfm_asia/lastfm_asia.edgelist',
        'label_file': 'lastfm_asia/lastfm_asia_labels.txt',
        'description': 'LastFM Asia social network (7624 nodes, 27806 edges)'
    }
}

# Algorithm parameter configuration
ALGORITHM_CONFIG = {
    'original': {
        'walk_length': 40,
        'num_walks': 10,
        'embed_size': 64,
        'window_size': 5,
        'workers': 2,
        'iter': 3,
        'opt1_reduce_len': True,
        'opt2_reduce_sim_calc': True
    },
    'graphlet': {
        'walk_length': 40,
        'num_walks': 10,
        'embed_size': 64,
        'window_size': 5,
        'workers': 2,
        'iter': 3,
        'max_layer': 3,
        'k': 5,  # graphlet size
        'distance_method': 'frobenius',
        'use_orbit_selection': False
    },
    'fusion': {
        'walk_length': 40,
        'num_walks': 10,
        'embed_size': 64,
        'window_size': 5,
        'workers': 2,
        'iter': 3,
        'alpha': 0.5,  # fusion weight
        'fusion_method': 'weighted'  # weighted, adaptive, confidence
    }
}

# Evaluation configuration
EVALUATION_CONFIG = {
    'train_ratio': 0.8,
    'random_state': 42,
    'classifiers': ['logistic', 'svm', 'rf'],
    'metrics': ['accuracy', 'f1_micro', 'f1_macro']
}

# External tools paths
EXTERNAL_TOOLS = {
    'orca_path': PARENT_ROOT / 'orca',
    'graph_embedding_path': PARENT_ROOT / 'GraphEmbedding'
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': PROJECT_ROOT / 'output' / 'logs' / 'struc2vec.log'
}

# Ensure directories exist
def ensure_directories():
    """Ensure all necessary directories exist"""
    for path in DATA_CONFIG.values():
        path.mkdir(parents=True, exist_ok=True)
    
    # Create log directory
    LOGGING_CONFIG['log_file'].parent.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ensure_directories()
    print("✅ Directory structure created")
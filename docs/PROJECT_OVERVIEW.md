# Graphlet Struc2Vec Project Overview

## 🎯 Project Description

This project implements and evaluates advanced graph embedding techniques, specifically focusing on **Enhanced Struc2Vec** methods that integrate graphlet-based structural features for improved node classification performance. The project combines traditional Struc2Vec algorithms with novel fusion methods to achieve superior embedding quality.

## 🏗️ Project Architecture

### Core Components

```
graphlet_struc2vec/
├── advancedStruc2vec/           # Main implementation directory
│   ├── src/                     # Source code
│   │   ├── algorithms/          # Core algorithms
│   │   │   ├── advanced_fusion_methods.py
│   │   │   ├── fusion_struc2vec.py
│   │   │   └── graphlet_struc2vec.py
│   │   └── utils/               # Utility functions
│   ├── config/                  # Configuration files
│   ├── data/                    # Processed datasets
│   ├── visualization_and_experiments/ # Experiment scripts
│   └── run_advanced_fusion_comparison.py
├── libs/                        # External libraries
│   ├── GraphEmbedding/         # Modified GraphEmbedding library
│   └── orca/                   # ORCA graphlet counting tool
├── data/                       # Raw datasets
│   ├── flight/                 # Airport networks
│   ├── wiki/                   # Wiki networks
│   └── lastfm_asia/            # Social networks
├── draw/                       # Visualization and plotting tools
└── docs/                       # Project documentation
```

## 🔬 Key Algorithms

### 1. Original Struc2Vec
- Baseline structural embedding method
- Captures structural similarity between nodes
- Uses structural distance measures

### 2. Enhanced Struc2Vec (Advanced Fusion)
- **Multi-Head Attention Fusion**: Transformer-style attention for optimal feature fusion
- **Adaptive Weighted Combination**: Dynamic weight adjustment based on graph properties
- **Graph Neural Fusion**: GNN-based feature integration
- **Spectral Fusion**: Spectral analysis-based combination methods

### 3. Pure Graphlet Struc2Vec
- Integration of graphlet counting with Struc2Vec
- Uses ORCA algorithm for precise graphlet enumeration
- Combines structural and graphlet-based features

## 📊 Datasets

The project evaluates algorithms on multiple real-world networks:

### Flight Networks
- **USA Airports**: 1,190 nodes, 13,599 edges, 4 categories
- **Brazil Airports**: Network of Brazilian airports
- **Europe Airports**: European airport connections

### Social Networks
- **LastFM Asia**: Asian user social network
- **Wiki**: Wikipedia page link network

### Key Dataset Statistics
- Node counts: 500-5,000+ nodes
- Edge counts: 2,000-15,000+ edges
- Categories: 4-30+ classes
- Network types: Transportation, social, information

## 🎯 Performance Results

### Best Experimental Results (USA Airports Dataset)

| Method | Test Accuracy | F1-Macro | CV Accuracy | Training Time |
|--------|---------------|----------|-------------|---------------|
| Original Struc2Vec | 48.46% | 47.92% | 48.91% | 0.08s |
| **Enhanced Struc2Vec** | **59.38%** | **59.28%** | **60.08%** | 0.27s |
| **Improvement** | **+10.92%** | **+11.36%** | **+11.18%** | **+0.19s** |

### Key Achievements
- ✅ **22-24% relative performance improvement**
- ✅ **Significantly better PCA visualization** (70.3% vs 39.7% information retention)
- ✅ **Reasonable computational efficiency** (minimal time cost for major performance gain)

## 🔧 Technical Stack

### Core Dependencies
- **Python 3.8+**: Main programming language
- **NetworkX**: Graph manipulation and analysis
- **NumPy/SciPy**: Numerical computations
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib/Seaborn**: Data visualization

### Specialized Libraries
- **GraphEmbedding**: Modified for Struc2Vec implementation
- **ORCA**: Graphlet counting algorithm
- **TensorFlow**: Deep learning components
- **Gensim**: Word2Vec-style embedding training

### Analysis Tools
- **PCA/t-SNE**: Dimensionality reduction
- **Spectral Clustering**: Graph clustering
- **Cross-validation**: Model evaluation

## 🚀 Quick Start

### Basic Usage
```bash
# Run advanced fusion comparison
cd advancedStruc2vec
python run_advanced_fusion_comparison.py --dataset usa-airports

# Run visualization experiments
cd visualization_and_experiments
python FIXED_REAL_EXPERIMENT.py
```

### Configuration
- Edit `advancedStruc2vec/config/config.py` for algorithm parameters
- Modify dataset paths in experiment scripts
- Adjust visualization settings in plotting functions

## 📈 Research Contributions

### 1. Advanced Fusion Methods
- **Multi-Head Attention**: Applies Transformer attention to graph embeddings
- **Adaptive Weighting**: Dynamic fusion weight optimization
- **Spectral Integration**: Eigenvalue-based feature combination

### 2. Comprehensive Evaluation Framework
- **Multiple datasets**: Diverse network types and sizes
- **Robust metrics**: Accuracy, F1-score, cross-validation
- **Visualization analysis**: PCA, t-SNE embedding quality assessment

### 3. Practical Implementation
- **Production-ready code**: Modular, extensible architecture
- **Experiment reproducibility**: Detailed parameter tracking
- **Performance optimization**: Efficient algorithm implementations

## 🎯 Use Cases

### Academic Research
- Graph embedding method development
- Structural similarity analysis
- Network node classification benchmarking

### Industrial Applications
- Social network analysis
- Transportation network optimization
- Knowledge graph embedding
- Recommendation systems

### Educational Purposes
- Graph theory visualization
- Machine learning algorithm comparison
- Network analysis methodology demonstration

## 🔮 Future Directions

### Algorithm Enhancement
- **Dynamic fusion weights**: Real-time weight adjustment
- **Multi-scale features**: Combining local and global structural patterns
- **Attention mechanisms**: More sophisticated attention models

### Scalability Improvements
- **Large graph handling**: Optimization for networks with 100K+ nodes
- **Distributed computing**: Multi-machine parallel processing
- **Memory efficiency**: Reduced memory footprint algorithms

### Application Extensions
- **Temporal networks**: Time-evolving graph analysis
- **Heterogeneous graphs**: Multi-type node and edge networks
- **Domain adaptation**: Cross-domain embedding transfer

## 📚 Documentation Structure

- `PROJECT_OVERVIEW.md` (this file): Overall project introduction
- `ADVANCED_STRUC2VEC_GUIDE.md`: Detailed implementation and usage guide
- Algorithm-specific documentation in source code
- Experiment results and analysis reports

## 🤝 Contributing

The project follows standard open-source contribution practices:
- Code follows PEP 8 style guidelines
- Comprehensive documentation for all methods
- Unit tests for core functionality
- Reproducible experiment scripts

## 📄 License and Citation

This project builds upon existing graph embedding research and libraries. Please cite relevant papers when using this work in academic publications.
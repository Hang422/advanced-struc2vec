# Advanced Graphlet-Enhanced Struc2Vec

## Project Overview

This is an enhanced version of Struc2Vec that integrates graphlet features and multiple advanced feature fusion methods.

## Core Features

- **Original Struc2Vec Algorithm**: Structural similarity based on degree sequences
- **Graphlet-enhanced Version**: Using ORCA tool to compute graphlet features
- **5 Advanced Fusion Methods**: 
  - Multi-head Attention Fusion
  - Hierarchical Pyramid Fusion
  - Spectral-aware Fusion
  - Community-aware Fusion
  - Ensemble Fusion

## Project Structure

```
graphlet_struc2vec/
├── finalVersion/           # Main project directory
│   ├── run_advanced_fusion_comparison.py  # 🚀 Main entry script
│   ├── src/               # Core algorithm implementations
│   ├── data/              # Datasets
│   ├── config/            # Configuration files
│   └── scripts/           # Utility scripts
├── libs/                  # Dependencies
│   ├── GraphEmbedding/    # Graph embedding library
│   └── orca/              # ORCA graphlet tool
├── data/                  # Datasets
└── README.md             # Project documentation
```

## Quick Start

```bash
# Basic comparison
cd finalVersion
python run_advanced_fusion_comparison.py --dataset europe-airports

# Advanced fusion methods comparison
python run_advanced_fusion_comparison.py --dataset europe-airports --fusion-methods attention,spectral,community,ensemble

# Custom parameters
python run_advanced_fusion_comparison.py --dataset wiki --fusion-methods attention --num-walks 10 --walk-length 50
```

## Supported Datasets

- **europe-airports**: European airport network
- **wiki**: Wikipedia network
- **lastfm**: Last.fm music network

## Dependencies

- Python 3.7+
- NetworkX
- NumPy
- scikit-learn
- ORCA (included)

## Algorithm Description

### Fusion Methods Comparison

| Method | Features | Use Case |
|------|------|----------|
| Attention Fusion | Adaptive weight learning | Complex heterogeneous graphs |
| Pyramid Fusion | Multi-scale information processing | Hierarchical structures |
| Spectral-aware Fusion | Based on graph spectral properties | Networks with clear community structure |
| Community-aware Fusion | Differentiation within/between communities | Social networks |
| Ensemble Fusion | Multi-method combination | Pursuing best performance |

## Citation

If this project helps your research, please consider citing:

## License

MIT License

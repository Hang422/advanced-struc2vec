# Advanced Struc2Vec Usage Manual

## 📋 Table of Contents

1. [Main Execution Scripts](#main-execution-scripts)
2. [Visualization and Experiment Scripts](#visualization-and-experiment-scripts)
3. [Configuration Files](#configuration-files)
4. [Dataset Management](#dataset-management)
5. [Common Usage Patterns](#common-usage-patterns)
6. [Troubleshooting Guide](#troubleshooting-guide)

---

## 🚀 Main Execution Scripts

### 1. `run_advanced_fusion_comparison.py`

**Purpose**: The primary script for comparing different fusion methods with extensive parameter control.

#### Basic Usage
```bash
cd advancedStruc2vec
python run_advanced_fusion_comparison.py [OPTIONS]
```

#### Command Line Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--dataset` | str | `usa-airports` | Dataset selection |
| `--fusion-methods` | str | `attention,pyramid,spectral,community,ensemble,pure-graphlet` | Comma-separated fusion methods |
| `--include-baseline` | flag | `True` | Include baseline Struc2Vec for comparison |
| `--num-walks` | int | `6` | Number of random walks per node |
| `--walk-length` | int | `30` | Length of each random walk |
| `--embed-size` | int | `64` | Embedding dimension size |
| `--iter` | int | `2` | Number of training iterations |
| `--workers` | int | `1` | Number of parallel workers |
| `--max-layer` | int | `3` | Maximum structural layers to consider |
| `--distance-method` | str | `frobenius` | Distance calculation method |
| `--num-runs` | int | `1` | Number of experiment runs for averaging |

#### Available Datasets
- `usa-airports`: USA airports network (1,190 nodes, 4 classes) **[Recommended]**
- `brazil-airports`: Brazil airports network
- `europe-airports`: Europe airports network  
- `wiki`: Wikipedia page network
- `lastfm`: LastFM Asia social network

#### Available Fusion Methods
- `attention`: Multi-head attention fusion
- `pyramid`: Hierarchical pyramid fusion
- `spectral`: Spectral analysis-based fusion
- `community`: Community structure-aware fusion
- `ensemble`: Ensemble method combination
- `pure-graphlet`: Pure graphlet-based approach

#### Distance Methods
- `frobenius`: Frobenius norm-based distance
- `eigenvalue`: Eigenvalue-based distance
- `trace`: Matrix trace-based distance

#### Example Commands

**Quick Test with Default Settings:**
```bash
python run_advanced_fusion_comparison.py
```

**Full Comparison on USA Airports:**
```bash
python run_advanced_fusion_comparison.py \
  --dataset usa-airports \
  --fusion-methods attention,pyramid,spectral,ensemble,pure-graphlet \
  --include-baseline \
  --num-runs 3 \
  --workers 4
```

**High-Quality Embeddings (Slower but Better Results):**
```bash
python run_advanced_fusion_comparison.py \
  --dataset usa-airports \
  --num-walks 80 \
  --walk-length 40 \
  --embed-size 128 \
  --iter 5 \
  --workers 8
```

**Fast Testing (Reduced Parameters):**
```bash
python run_advanced_fusion_comparison.py \
  --dataset usa-airports \
  --fusion-methods attention,spectral \
  --num-walks 4 \
  --walk-length 20 \
  --embed-size 32 \
  --iter 1
```

#### Output Format
The script generates detailed results including:
- **Accuracy scores** for each method
- **F1-macro and F1-micro scores**
- **Training time comparisons**
- **Success/failure status for each method**
- **Average results** (when `--num-runs > 1`)

---

## 🔬 Visualization and Experiment Scripts

### 2. `FIXED_REAL_EXPERIMENT.py` ⭐ **[RECOMMENDED]**

**Purpose**: Production-ready experiment with comprehensive evaluation and visualization.

#### Usage
```bash
cd advancedStruc2vec/visualization_and_experiments
python FIXED_REAL_EXPERIMENT.py
```

#### Key Features
- ✅ **Real algorithm implementation**: Uses actual GraphEmbedding library
- ✅ **Comprehensive evaluation**: Multiple metrics and cross-validation
- ✅ **Publication-quality visualizations**: PCA plots, performance charts
- ✅ **Robust error handling**: Graceful handling of failures
- ✅ **Detailed reporting**: Comprehensive result summaries

#### Generated Output Files
```
fixed_real_results/
├── experiment_summary.txt      # Detailed text report
├── performance_comparison.png  # Performance metrics chart
├── pca_comparison.png         # PCA visualization comparison
└── timing_comparison.png      # Training time analysis
```

#### Configuration (Internal)
The script uses hardcoded optimized parameters:
```python
CONFIG = {
    'dataset': 'usa-airports',
    'walk_length': 10,
    'num_walks': 80,
    'dimensions': 64,
    'window': 10,
    'workers': 1,
    'iter': 5
}
```

### 3. `ADVANCED_EXPERIMENTS.py`

**Purpose**: Advanced experimental framework with multiple evaluation strategies.

#### Usage
```bash
python ADVANCED_EXPERIMENTS.py
```

#### Features
- **Multi-dataset support**: Tests across different networks
- **Robustness evaluation**: Multiple random seeds
- **Scalability analysis**: Performance vs. graph size
- **Cross-validation**: K-fold evaluation
- **Statistical analysis**: Confidence intervals and significance tests

### 4. `USA_AIRPORTS_EXPERIMENT.py`

**Purpose**: Comprehensive analysis specifically for USA airports dataset.

#### Usage
```bash
python USA_AIRPORTS_EXPERIMENT.py
```

#### Specialized Features
- **Multiple classifiers**: Logistic Regression, Random Forest, SVM
- **Advanced visualizations**: t-SNE, UMAP projections
- **Detailed metrics**: Precision, Recall, F1-score per class
- **Confusion matrix analysis**: Class-specific performance
- **Geographic insights**: Airport-specific analysis

### 5. `USA_AIRPORTS_SIMPLE.py`

**Purpose**: Simplified version for quick testing and prototyping.

#### Usage
```bash
python USA_AIRPORTS_SIMPLE.py
```

#### Use Cases
- **Quick validation**: Fast method verification
- **Parameter tuning**: Rapid iteration for optimization
- **Debugging**: Minimal setup for troubleshooting
- **Educational**: Clean, simple example implementation

### 6. `USA_AIRPORTS_QUICK_TEST.py`

**Purpose**: Ultra-fast testing with minimal parameters.

#### Usage
```bash
python USA_AIRPORTS_QUICK_TEST.py
```

#### Features
- **Reduced parameters**: Fastest execution time
- **Basic comparison**: Essential metrics only
- **Development testing**: Quick sanity checks
- **CI/CD integration**: Automated testing pipeline

### 7. `VISUALIZATION_EXPERIMENTS.py`

**Purpose**: Comprehensive visualization toolkit for embedding analysis.

#### Usage
```bash
python VISUALIZATION_EXPERIMENTS.py
```

#### Visualization Types
- **PCA Analysis**: Dimensional reduction and variance explanation
- **t-SNE Projections**: Nonlinear embedding visualization
- **UMAP Plots**: Uniform Manifold Approximation
- **Similarity Heatmaps**: Node-to-node similarity matrices
- **Network Structure**: Graph topology visualization
- **Clustering Analysis**: Community detection visualization

#### Interactive Features
- **Plotly Integration**: Interactive plots with zoom/pan
- **Multi-scale Analysis**: Different embedding dimensions
- **Comparative Views**: Side-by-side method comparison
- **Export Options**: High-resolution image output

---

## ⚙️ Configuration Files

### `advancedStruc2vec/config/config.py`

Central configuration file for algorithm parameters:

```python
# Core Struc2Vec Parameters
STRUC2VEC_CONFIG = {
    'walk_length': 10,          # Random walk length
    'num_walks': 80,            # Number of walks per node
    'dimensions': 64,           # Embedding dimensions
    'window_size': 10,          # Word2Vec window size
    'workers': 4,              # Parallel workers
    'iter': 5,                 # Training iterations
    'stay_prob': 0.3,          # Stay probability for walks
    'opt1_reduce_len': True,   # Optimization 1: reduce length
    'opt2_reduce_sim_calc': True, # Optimization 2: reduce similarity calc
}

# Advanced Fusion Parameters
FUSION_CONFIG = {
    'attention_heads': 4,       # Multi-head attention heads
    'pyramid_levels': 3,        # Hierarchical pyramid levels
    'spectral_components': 10,  # Spectral fusion components
    'fusion_alpha': 0.5,       # Adaptive fusion weight
    'community_resolution': 1.0, # Community detection resolution
}

# Dataset Paths
DATA_PATHS = {
    'base_dir': 'data/raw/data',
    'flight_dir': 'flight',
    'social_dir': 'lastfm_asia', 
    'wiki_dir': 'wiki',
}

# Evaluation Settings
EVAL_CONFIG = {
    'test_size': 0.3,          # Train-test split ratio
    'cv_folds': 5,             # Cross-validation folds
    'random_seed': 42,         # Reproducibility seed
    'max_iter': 1000,          # Classifier max iterations
}

# Output Settings
OUTPUT_CONFIG = {
    'save_embeddings': True,    # Save embedding files
    'save_plots': True,        # Save visualization plots
    'plot_format': 'png',      # Output image format
    'plot_dpi': 300,          # Image resolution
}
```

---

## 📁 Dataset Management

### Dataset Structure
```
data/
├── flight/                    # Aviation networks
│   ├── usa-airports.edgelist
│   ├── labels-usa-airports.txt
│   ├── brazil-airports.edgelist
│   ├── labels-brazil-airports.txt
│   └── europe-airports.edgelist
├── wiki/                      # Wikipedia networks
│   ├── Wiki_edgelist.txt
│   └── wiki_labels.txt
└── lastfm_asia/              # Social networks
    ├── lastfm_asia.edgelist
    └── lastfm_asia_labels.txt
```

### Dataset Loading Functions

Most scripts include automatic dataset loading:

```python
def load_dataset(dataset_name):
    """Automatic dataset loader with validation"""
    datasets = {
        'usa-airports': {
            'graph': 'data/flight/usa-airports.edgelist',
            'labels': 'data/flight/labels-usa-airports.txt',
            'nodes': 1190, 'edges': 13599, 'classes': 4
        },
        # ... other datasets
    }
    return load_graph_and_labels(datasets[dataset_name])
```

### Custom Dataset Integration

To add a new dataset:

1. **Add data files** to appropriate directory
2. **Update dataset configuration** in scripts
3. **Verify format compatibility**:
   - **Graph file**: Edge list format `node1 node2`
   - **Label file**: Tab/space separated `node_id label`

---

## 🎯 Common Usage Patterns

### Pattern 1: Quick Method Comparison
```bash
# Compare top 3 methods on default dataset
python run_advanced_fusion_comparison.py \
  --fusion-methods attention,spectral,ensemble \
  --num-runs 3
```

### Pattern 2: Production-Quality Results
```bash
# Generate publication-ready results
cd visualization_and_experiments
python FIXED_REAL_EXPERIMENT.py
```

### Pattern 3: Parameter Optimization
```bash
# Test different embedding dimensions
for dim in 32 64 128; do
  python run_advanced_fusion_comparison.py \
    --embed-size $dim \
    --fusion-methods attention \
    --num-runs 1
done
```

### Pattern 4: Multi-Dataset Analysis
```bash
# Test across all datasets
for dataset in usa-airports brazil-airports wiki; do
  python run_advanced_fusion_comparison.py \
    --dataset $dataset \
    --fusion-methods attention,spectral
done
```

### Pattern 5: High-Performance Computing
```bash
# Maximize performance with parallel processing
python run_advanced_fusion_comparison.py \
  --workers 16 \
  --num-walks 200 \
  --walk-length 80 \
  --embed-size 256 \
  --iter 10
```

---

## 🛠️ Troubleshooting Guide

### Common Issues and Solutions

#### 1. Memory Issues
**Problem**: `MemoryError` or system freeze
**Solutions**:
```bash
# Reduce parameters
python run_advanced_fusion_comparison.py \
  --num-walks 4 \
  --walk-length 10 \
  --embed-size 32 \
  --workers 1

# Use optimizations
# Edit config.py:
STRUC2VEC_CONFIG['opt1_reduce_len'] = True
STRUC2VEC_CONFIG['opt2_reduce_sim_calc'] = True
```

#### 2. Import Errors
**Problem**: `ModuleNotFoundError` or import issues
**Solutions**:
```bash
# Check Python path
export PYTHONPATH="/path/to/graphlet_struc2vec:$PYTHONPATH"

# Install missing dependencies
pip install -r requirements.txt

# Verify libs installation
ls libs/GraphEmbedding/ge/models/
```

#### 3. Performance Issues
**Problem**: Very slow execution
**Solutions**:
```bash
# Use fast parameters
python run_advanced_fusion_comparison.py \
  --num-walks 4 \
  --walk-length 20 \
  --iter 1 \
  --workers 4

# Use simple experiment
python USA_AIRPORTS_QUICK_TEST.py
```

#### 4. Visualization Errors
**Problem**: Matplotlib or plotting errors
**Solutions**:
```bash
# Set backend
export MPLBACKEND=Agg

# Or modify script:
import matplotlib
matplotlib.use('Agg')
```

#### 5. Dataset Loading Errors
**Problem**: File not found or format errors
**Solutions**:
```bash
# Check file paths
ls -la data/flight/usa-airports.edgelist
ls -la data/flight/labels-usa-airports.txt

# Verify format
head -5 data/flight/usa-airports.edgelist
head -5 data/flight/labels-usa-airports.txt
```

### Performance Optimization Tips

#### 1. Parameter Tuning for Speed vs Quality

**Fast Testing** (Development):
```python
CONFIG_FAST = {
    'num_walks': 4,
    'walk_length': 10,
    'embed_size': 32,
    'iter': 1,
    'workers': 2
}
```

**Balanced** (Default):
```python
CONFIG_BALANCED = {
    'num_walks': 6,
    'walk_length': 30,
    'embed_size': 64,
    'iter': 2,
    'workers': 4
}
```

**High Quality** (Production):
```python
CONFIG_HIGH_QUALITY = {
    'num_walks': 80,
    'walk_length': 40,
    'embed_size': 128,
    'iter': 5,
    'workers': 8
}
```

#### 2. Memory Management

**Large Graphs** (>5000 nodes):
```python
# Enable all optimizations
CONFIG_LARGE = {
    'opt1_reduce_len': True,
    'opt2_reduce_sim_calc': True,
    'opt3_num_layers': 2,  # Reduce layers
    'workers': 2,          # Reduce parallel workers
    'embed_size': 64,      # Moderate embedding size
}
```

#### 3. Parallel Processing

**Multi-core Systems**:
```bash
# Use all available cores (adjust based on your system)
python run_advanced_fusion_comparison.py \
  --workers $(nproc) \
  --fusion-methods attention
```

### Debug Mode

Enable verbose output for debugging:
```python
# In script, modify:
model = Struc2Vec(G, verbose=40)  # Enable verbose output
```

### Logging Configuration

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
```

---

## 📊 Expected Results and Benchmarks

### USA Airports Dataset Benchmarks

**Hardware**: Standard laptop (8GB RAM, 4 cores)

| Configuration | Time | Accuracy | Memory |
|---------------|------|----------|---------|
| Fast | ~30s | ~52% | ~2GB |
| Balanced | ~2min | ~57% | ~4GB |
| High Quality | ~10min | ~60% | ~6GB |

### Success Criteria

**Minimum Expected Results**:
- **Baseline Struc2Vec**: 45-50% accuracy
- **Enhanced Methods**: 55-62% accuracy
- **Training Time**: 0.1-10 minutes (depending on parameters)

**Red Flags** (Indicating Issues):
- Accuracy < 40%: Check dataset loading
- Training time > 30min: Reduce parameters
- Memory usage > 16GB: Enable optimizations

---

This usage manual provides comprehensive guidance for all aspects of the Advanced Struc2Vec project. For additional support, refer to the code comments and error messages, which provide specific guidance for individual use cases.
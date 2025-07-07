# Graphlet-Enhanced Struc2Vec Final Version

这是 Graphlet 增强 Struc2Vec 算法的最终重构版本，提供了清晰的项目结构和统一的测试接口。

## 项目结构

```
finalVersion/
├── README.md                    # 项目说明
├── requirements.txt             # 依赖包
├── config/                      # 配置文件
│   └── config.py               # 全局配置
├── data/                        # 原始数据
│   ├── processed/              # 预处理后的数据
│   └── raw/                    # 原始数据集
├── src/                         # 源代码
│   ├── __init__.py
│   ├── algorithms/             # 算法实现
│   │   ├── __init__.py
│   │   ├── base.py            # 基础算法接口
│   │   ├── original_struc2vec.py  # 原始算法
│   │   ├── graphlet_struc2vec.py  # Graphlet增强版
│   │   └── fusion_struc2vec.py    # 融合版本
│   ├── utils/                  # 工具函数
│   │   ├── __init__.py
│   │   ├── data_loader.py     # 数据加载
│   │   ├── evaluator.py       # 评估工具
│   │   └── fusion.py          # 融合策略
│   └── graphlet/               # Graphlet相关
│       ├── __init__.py
│       ├── orca_wrapper.py    # ORCA工具包装
│       └── distance_computer.py # 距离计算
├── temp/                        # 临时文件
├── output/                      # 输出结果
│   ├── distances/              # 距离文件
│   ├── embeddings/             # 嵌入文件
│   └── results/                # 评估结果
├── scripts/                     # 运行脚本
│   ├── run_comparison.py       # 统一测试脚本
│   ├── generate_distances.py   # 生成距离文件
│   └── optimize_fusion.py      # 融合优化
└── tests/                       # 测试文件
    ├── __init__.py
    └── test_algorithms.py      # 算法测试
```

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行基础比较
```bash
python scripts/run_comparison.py --dataset brazil-airports --methods original,graphlet,fusion
```

### 3. 生成距离文件
```bash
python scripts/generate_distances.py --dataset brazil-airports
```

### 4. 优化融合参数
```bash
python scripts/optimize_fusion.py --dataset brazil-airports
```

## 算法说明

### 原始 Struc2Vec
- 基于节点度序列的结构相似性
- 使用动态时间规整(DTW)计算距离

### Graphlet 增强版
- 使用 ORCA 计算 Graphlet 度向量(GDV)
- 支持多种距离度量方法
- 自适应特征选择

### 融合版本
- 加权融合原始距离和 Graphlet 距离
- 支持多种融合策略：
  - 线性加权融合
  - 自适应融合
  - 置信度融合

## 配置说明

所有配置都在 `config/config.py` 中统一管理，包括：
- 数据路径
- 算法参数
- 输出设置
- 临时文件管理
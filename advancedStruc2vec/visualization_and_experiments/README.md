# 可视化与实验模块

本目录包含所有与可视化分析和实验评估相关的代码和结果文件。

## 📁 目录结构

### 🔬 **核心实验脚本**

#### **框架性实验**
- `VISUALIZATION_EXPERIMENTS.py` - 通用可视化实验框架
- `ADVANCED_EXPERIMENTS.py` - 高级实验框架 (鲁棒性、可扩展性等)

#### **实际算法对比实验**
- `REAL_ALGORITHMS_EXPERIMENT.py` - 真实算法对比实验 (最初版本)
- `FIXED_REAL_EXPERIMENT.py` - 修复版真实算法实验 ⭐ **推荐使用**

#### **USA机场数据集专项实验**
- `USA_AIRPORTS_EXPERIMENT.py` - 完整版USA机场实验
- `USA_AIRPORTS_SIMPLE.py` - 简化版USA机场实验
- `USA_AIRPORTS_QUICK_TEST.py` - 快速测试版本

### 📊 **实验结果目录**

#### **真实算法对比结果** ⭐
- `fixed_real_results/` - 最终成功的实验结果
  - `experiment_summary.txt` - 实验总结报告
  - `pca_comparison.png` - PCA可视化对比
  - `performance_comparison.png` - 性能对比图表
  - `timing_comparison.png` - 训练时间对比

#### **其他实验结果**
- `experimental_results/` - 框架性实验结果
- `usa_airports_results/` - USA机场实验结果
- `usa_results/` - USA简化实验结果  
- `visualizations/` - 可视化示例结果

## 🎯 **关键实验成果**

### **最佳实验结果** (from `FIXED_REAL_EXPERIMENT.py`)

**数据集**: USA Airports (1190节点, 13599边, 4类)

| 方法 | 测试准确率 | F1-宏平均 | CV准确率 | 训练时间 |
|------|-----------|----------|----------|----------|
| Original Struc2Vec | 48.46% | 47.92% | 48.91% | 0.08s |
| **Enhanced Struc2Vec** | **59.38%** | **59.28%** | **60.08%** | 0.27s |
| **提升** | **+10.92%** | **+11.36%** | **+11.18%** | **+0.19s** |

**关键发现**:
- ✅ **22-24%相对性能提升**
- ✅ **PCA可视化显著改善** (信息保持70.3% vs 39.7%)
- ✅ **计算效率合理** (微小时间成本换取巨大性能提升)

## 🚀 **快速开始**

### **运行最佳实验**
```bash
# 运行真实算法对比实验
python FIXED_REAL_EXPERIMENT.py

# 结果将保存在 fixed_real_results/ 目录
```

### **运行框架性实验**
```bash
# 运行可视化实验
python VISUALIZATION_EXPERIMENTS.py

# 运行高级实验框架
python ADVANCED_EXPERIMENTS.py
```

## 📋 **实验说明**

### **成功实验**: `FIXED_REAL_EXPERIMENT.py`
- ✅ 使用真实的Struc2Vec算法库
- ✅ 成功处理USA airports大规模数据
- ✅ 实现了Enhanced Struc2Vec vs Original Struc2Vec对比
- ✅ 生成了完整的可视化和性能报告

### **框架实验**: `VISUALIZATION_EXPERIMENTS.py` & `ADVANCED_EXPERIMENTS.py`
- 🔧 提供通用的实验框架
- 🔧 适用于各种数据集和算法
- 🔧 包含丰富的可视化工具

### **专项实验**: `USA_AIRPORTS_*.py`
- 🎯 专门针对USA airports数据集
- 🎯 不同复杂度版本 (完整版、简化版、快速版)
- 🎯 作为方法验证和调试工具

## 📊 **可视化功能**

### **PCA分析**
- 降维可视化 (高维嵌入 → 2D图)
- 聚类质量评估
- 类别分离度分析

### **性能对比**
- 多指标对比图表
- 训练时间分析
- 交叉验证结果

### **嵌入质量**
- 结构相似性热力图
- 距离分布分析
- 相关性评估

## 🛠 **技术栈**

- **核心算法**: Struc2Vec (GraphEmbedding库)
- **机器学习**: scikit-learn
- **可视化**: matplotlib, seaborn, plotly
- **数据处理**: numpy, pandas, networkx
- **降维**: PCA, t-SNE, UMAP

## 📈 **实验价值**

1. **方法验证**: 证明了Enhanced Struc2Vec的有效性
2. **性能基准**: 建立了在真实网络上的性能基准
3. **可视化洞察**: 通过PCA分析理解算法改进原理
4. **框架贡献**: 提供了可复用的实验评估框架

## 🔄 **使用建议**

### **研究用途**:
- 使用 `FIXED_REAL_EXPERIMENT.py` 获取最佳结果
- 参考 `fixed_real_results/` 中的实验报告

### **开发用途**:
- 使用 `VISUALIZATION_EXPERIMENTS.py` 进行方法调试
- 使用 `ADVANCED_EXPERIMENTS.py` 进行全面评估

### **快速验证**:
- 使用 `USA_AIRPORTS_SIMPLE.py` 进行快速测试
- 适合方法原型验证和参数调优
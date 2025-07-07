#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试随机种子对结果的影响
验证数据泄露问题
"""
import sys
import numpy as np
from pathlib import Path

# 添加父项目路径
parent_path = Path(__file__).parent.parent
sys.path.append(str(parent_path))

from GraphEmbedding.ge.classify import read_node_label, Classifier
from sklearn.linear_model import LogisticRegression

def test_seed_impact():
    """测试不同随机种子对数据分割的影响"""
    
    # 加载数据
    data_base = parent_path / 'data'
    label_file = data_base / 'flight/labels-brazil-airports.txt'
    X, Y = read_node_label(str(label_file), skip_head=True)
    
    print("🧪 随机种子影响测试")
    print("=" * 50)
    print(f"数据集: brazil-airports")
    print(f"总样本数: {len(X)}")
    print(f"训练比例: 80%")
    
    # 测试不同种子的分割结果
    seeds = [0, 42, 123, 999]
    
    for seed in seeds:
        print(f"\\n🎲 随机种子: {seed}")
        
        # 模拟原版的分割方式
        np.random.seed(seed)
        training_size = int(0.8 * len(X))
        shuffle_indices = np.random.permutation(np.arange(len(X)))
        
        train_indices = shuffle_indices[:training_size]
        test_indices = shuffle_indices[training_size:]
        
        X_train = [X[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]
        Y_train = [Y[i] for i in train_indices]
        Y_test = [Y[i] for i in test_indices]
        
        print(f"   训练集: {len(X_train)} 样本")
        print(f"   测试集: {len(X_test)} 样本")
        print(f"   训练集前3个节点: {X_train[:3]}")
        print(f"   测试集前3个节点: {X_test[:3]}")
        
        # 检查标签分布
        train_labels = [y[0] if isinstance(y, list) else y for y in Y_train]
        test_labels = [y[0] if isinstance(y, list) else y for y in Y_test]
        
        train_unique = set(train_labels)
        test_unique = set(test_labels)
        
        print(f"   训练集标签种类: {train_unique}")
        print(f"   测试集标签种类: {test_unique}")
        print(f"   标签重叠: {train_unique & test_unique}")
        
        # 检查是否有测试集独有的标签
        test_only = test_unique - train_unique
        if test_only:
            print(f"   ⚠️  测试集独有标签: {test_only}")
        else:
            print(f"   ✅ 无测试集独有标签")

def simulate_original_vs_safe():
    """模拟原版和安全版的评估差异"""
    
    print("\\n\\n🔬 原版 vs 安全版 模拟对比")
    print("=" * 50)
    
    # 加载数据
    data_base = parent_path / 'data'
    label_file = data_base / 'flight/labels-brazil-airports.txt'
    X, Y = read_node_label(str(label_file), skip_head=True)
    
    # 创建模拟嵌入（随机向量）
    np.random.seed(12345)  # 固定嵌入生成
    embeddings = {}
    for node in X:
        embeddings[node] = np.random.randn(64)
    
    print(f"生成了 {len(embeddings)} 个随机嵌入向量")
    
    # 原版方式 (seed=0)
    print("\\n📊 原版方式 (seed=0):")
    try:
        clf_original = Classifier(embeddings=embeddings, clf=LogisticRegression(max_iter=1000))
        metrics_original = clf_original.split_train_evaluate(X, Y, 0.8, seed=0)
        print(f"   准确率: {metrics_original['acc']:.4f}")
        print(f"   F1-micro: {metrics_original['micro']:.4f}")
        print(f"   F1-macro: {metrics_original['macro']:.4f}")
    except Exception as e:
        print(f"   ❌ 原版评估失败: {e}")
    
    # 安全版方式 (seed=42)  
    print("\\n🛡️  安全版方式 (seed=42):")
    try:
        clf_safe = Classifier(embeddings=embeddings, clf=LogisticRegression(max_iter=1000))
        metrics_safe = clf_safe.split_train_evaluate(X, Y, 0.8, seed=42)
        print(f"   准确率: {metrics_safe['acc']:.4f}")
        print(f"   F1-micro: {metrics_safe['micro']:.4f}")
        print(f"   F1-macro: {metrics_safe['macro']:.4f}")
    except Exception as e:
        print(f"   ❌ 安全版评估失败: {e}")
    
    # 同种子对比 (都用seed=42)
    print("\\n🎯 同种子对比 (都用seed=42):")
    try:
        clf_same = Classifier(embeddings=embeddings, clf=LogisticRegression(max_iter=1000))
        metrics_same = clf_same.split_train_evaluate(X, Y, 0.8, seed=42)
        print(f"   准确率: {metrics_same['acc']:.4f}")
        print(f"   F1-micro: {metrics_same['micro']:.4f}")
        print(f"   F1-macro: {metrics_same['macro']:.4f}")
        
        if abs(metrics_safe['acc'] - metrics_same['acc']) < 0.001:
            print("   ✅ 与安全版结果一致 - 说明主要差异来自随机种子")
        else:
            print("   ⚠️  与安全版结果不一致 - 可能有其他因素")
            
    except Exception as e:
        print(f"   ❌ 同种子评估失败: {e}")

def analyze_data_leakage():
    """分析数据泄露的具体影响"""
    
    print("\\n\\n🕵️ 数据泄露影响分析")
    print("=" * 50)
    
    # 加载数据
    data_base = parent_path / 'data'
    label_file = data_base / 'flight/labels-brazil-airports.txt'
    X, Y = read_node_label(str(label_file), skip_head=True)
    
    print(f"原始标签格式示例:")
    for i in range(min(5, len(Y))):
        print(f"   Y[{i}] = {Y[i]} (类型: {type(Y[i])})")
    
    # 检查是否是多标签
    is_multilabel = any(isinstance(y, list) and len(y) > 1 for y in Y)
    print(f"\\n是否为多标签数据: {is_multilabel}")
    
    if is_multilabel:
        print("多标签统计:")
        all_labels = []
        for y in Y:
            if isinstance(y, list):
                all_labels.extend(y)
            else:
                all_labels.append(y)
        
        unique_labels = set(all_labels)
        print(f"   总标签种类数: {len(unique_labels)}")
        print(f"   所有标签: {sorted(unique_labels)}")
        
        # 检查标签分布
        from collections import Counter
        label_counts = Counter(all_labels)
        print(f"   标签分布: {dict(label_counts.most_common())}")
    else:
        # 单标签统计
        unique_labels = set(Y)
        print(f"标签种类数: {len(unique_labels)}")
        print(f"所有标签: {sorted(unique_labels)}")

if __name__ == "__main__":
    test_seed_impact()
    simulate_original_vs_safe()
    analyze_data_leakage()
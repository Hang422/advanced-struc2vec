#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
严格防止数据泄露的评估模块
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class SafeClassifier:
    """严格防止数据泄露的分类器"""
    
    def __init__(self, embeddings, clf=None):
        self.embeddings = embeddings
        self.clf = clf if clf is not None else LogisticRegression(max_iter=1000)
        self.label_encoder = LabelEncoder()
    
    def safe_split_train_evaluate(self, X, Y, train_ratio=0.8, random_seed=42):
        """
        严格的训练测试分离评估
        
        Args:
            X: 节点ID列表
            Y: 节点标签列表 (每个节点可能有多个标签)
            train_ratio: 训练集比例
            random_seed: 随机种子
            
        Returns:
            评估指标字典
        """
        np.random.seed(random_seed)
        
        # 处理多标签情况 - 取第一个标签作为主要标签
        Y_single = [labels[0] if isinstance(labels, list) and len(labels) > 0 else labels for labels in Y]
        
        # 随机打乱数据
        indices = np.random.permutation(len(X))
        train_size = int(len(X) * train_ratio)
        
        # 严格分离训练集和测试集
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        X_train = [X[i] for i in train_indices]
        Y_train = [Y_single[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]
        Y_test = [Y_single[i] for i in test_indices]
        
        # 只使用训练集的标签来fit label encoder
        self.label_encoder.fit(Y_train)
        
        # 检查测试集是否有新标签
        unique_test_labels = set(Y_test)
        unique_train_labels = set(Y_train)
        unknown_labels = unique_test_labels - unique_train_labels
        
        if unknown_labels:
            print(f"⚠️  警告: 测试集包含训练集中未见过的标签: {unknown_labels}")
            # 过滤掉未知标签的测试样本
            filtered_test = [(x, y) for x, y in zip(X_test, Y_test) if y in unique_train_labels]
            if len(filtered_test) == 0:
                return {'acc': 0.0, 'micro': 0.0, 'macro': 0.0, 'error': 'No valid test samples'}
            X_test, Y_test = zip(*filtered_test)
            X_test, Y_test = list(X_test), list(Y_test)
        
        # 转换标签为数值
        Y_train_encoded = self.label_encoder.transform(Y_train)
        Y_test_encoded = self.label_encoder.transform(Y_test)
        
        # 提取嵌入特征 - 只使用训练/测试节点的嵌入
        try:
            X_train_embeddings = np.array([self.embeddings[node] for node in X_train])
            X_test_embeddings = np.array([self.embeddings[node] for node in X_test])
        except KeyError as e:
            return {'acc': 0.0, 'micro': 0.0, 'macro': 0.0, 'error': f'Missing embedding for node: {e}'}
        
        # 训练分类器 - 只使用训练数据
        self.clf.fit(X_train_embeddings, Y_train_encoded)
        
        # 在测试集上预测
        Y_pred = self.clf.predict(X_test_embeddings)
        
        # 计算指标
        try:
            accuracy = accuracy_score(Y_test_encoded, Y_pred)
            f1_micro = f1_score(Y_test_encoded, Y_pred, average='micro')
            f1_macro = f1_score(Y_test_encoded, Y_pred, average='macro')
            
            return {
                'acc': accuracy,
                'micro': f1_micro,
                'macro': f1_macro,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'num_classes': len(unique_train_labels)
            }
        except Exception as e:
            return {'acc': 0.0, 'micro': 0.0, 'macro': 0.0, 'error': str(e)}

def safe_evaluate_method(embeddings, X, Y, method_name, random_seed=42):
    """
    安全的方法评估函数
    
    Args:
        embeddings: 节点嵌入字典
        X: 节点ID列表
        Y: 节点标签列表
        method_name: 方法名称
        random_seed: 随机种子
        
    Returns:
        评估结果字典
    """
    try:
        clf = SafeClassifier(embeddings)
        metrics = clf.safe_split_train_evaluate(X, Y, train_ratio=0.8, random_seed=random_seed)
        
        if 'error' in metrics:
            print(f"   ❌ {method_name} 评估失败: {metrics['error']}")
            return {
                'method': method_name,
                'accuracy': 0.0,
                'f1_micro': 0.0,
                'f1_macro': 0.0,
                'success': False,
                'error': metrics['error']
            }
        
        result = {
            'method': method_name,
            'accuracy': metrics['acc'],
            'f1_micro': metrics['micro'],
            'f1_macro': metrics['macro'],
            'success': True,
            'train_size': metrics['train_size'],
            'test_size': metrics['test_size'],
            'num_classes': metrics['num_classes']
        }
        
        print(f"   📊 {method_name}: 准确率={result['accuracy']:.4f}, 训练集={metrics['train_size']}, 测试集={metrics['test_size']}")
        return result
        
    except Exception as e:
        print(f"   ❌ {method_name} 评估异常: {e}")
        return {
            'method': method_name,
            'accuracy': 0.0,
            'f1_micro': 0.0,
            'f1_macro': 0.0,
            'success': False,
            'error': str(e)
        }

class DataLeakageChecker:
    """数据泄露检查器"""
    
    @staticmethod
    def check_train_test_overlap(X_train, X_test):
        """检查训练集和测试集是否有重叠"""
        train_set = set(X_train)
        test_set = set(X_test)
        overlap = train_set & test_set
        
        if overlap:
            print(f"❌ 数据泄露警告: 训练集和测试集有 {len(overlap)} 个重叠节点: {list(overlap)[:5]}...")
            return True
        else:
            print(f"✅ 无数据泄露: 训练集({len(train_set)})和测试集({len(test_set)})完全分离")
            return False
    
    @staticmethod
    def check_embedding_integrity(embeddings, X_all):
        """检查嵌入完整性"""
        missing_nodes = [node for node in X_all if node not in embeddings]
        if missing_nodes:
            print(f"⚠️  缺失嵌入的节点: {len(missing_nodes)} 个, 示例: {missing_nodes[:5]}")
            return False
        else:
            print(f"✅ 嵌入完整: 所有 {len(X_all)} 个节点都有嵌入")
            return True
    
    @staticmethod
    def validate_experimental_setup(embeddings, X, Y, train_ratio=0.8):
        """验证实验设置的合理性"""
        print("🔍 实验设置验证:")
        
        # 检查数据大小
        print(f"   总节点数: {len(X)}")
        print(f"   总标签数: {len(Y)}")
        print(f"   训练比例: {train_ratio}")
        print(f"   预期训练集大小: {int(len(X) * train_ratio)}")
        print(f"   预期测试集大小: {len(X) - int(len(X) * train_ratio)}")
        
        # 检查标签分布
        if isinstance(Y[0], list):
            all_labels = [label for labels in Y for label in labels]
        else:
            all_labels = Y
        
        unique_labels = set(all_labels)
        print(f"   唯一标签数: {len(unique_labels)}")
        
        # 检查是否有足够的样本进行有意义的评估
        min_test_size = max(len(unique_labels) * 2, 10)  # 每个类至少2个样本，或者至少10个样本
        actual_test_size = len(X) - int(len(X) * train_ratio)
        
        if actual_test_size < min_test_size:
            print(f"⚠️  警告: 测试集可能太小 ({actual_test_size}) 进行可靠评估")
            print(f"   建议测试集至少有 {min_test_size} 个样本")
        
        # 检查嵌入完整性
        DataLeakageChecker.check_embedding_integrity(embeddings, X)
        
        return True

if __name__ == "__main__":
    print("🛡️  安全评估模块加载完成")
    print("主要功能:")
    print("  - SafeClassifier: 严格防止数据泄露的分类器")
    print("  - safe_evaluate_method: 安全的方法评估函数")
    print("  - DataLeakageChecker: 数据泄露检查器")
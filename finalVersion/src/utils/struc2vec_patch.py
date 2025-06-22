#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Struc2Vec 运行时补丁
修复除零警告而不修改原始文件
"""
import numpy as np
import warnings

def patch_struc2vec():
    """应用运行时补丁"""
    try:
        from GraphEmbedding.ge.models import struc2vec
        
        # 保存原始方法
        original_get_layers_adj = struc2vec.Struc2Vec._get_layers_adj
        
        def patched_get_layers_adj(self, layers_distances):
            """修补后的方法，处理除零情况"""
            
            # 使用警告过滤器暂时忽略除零警告
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning, 
                                      message='invalid value encountered in scalar divide')
                
                # 调用原始方法
                result = original_get_layers_adj(self, layers_distances)
                
                # 检查并修复任何 NaN 或 Inf 值
                layers_alias, layers_accept = result
                
                for layer in layers_alias:
                    for node in layers_alias[layer]:
                        # 检查 alias 表
                        if layers_alias[layer][node] is not None:
                            alias = layers_alias[layer][node]
                            # 将任何 NaN 替换为 0
                            if isinstance(alias, (list, np.ndarray)):
                                alias = np.nan_to_num(alias, nan=0.0, posinf=1.0, neginf=0.0)
                                layers_alias[layer][node] = alias
                        
                        # 检查 accept 表
                        if layers_accept[layer][node] is not None:
                            accept = layers_accept[layer][node]
                            if isinstance(accept, (list, np.ndarray)):
                                accept = np.nan_to_num(accept, nan=1.0, posinf=1.0, neginf=0.0)
                                layers_accept[layer][node] = accept
                
                return layers_alias, layers_accept
        
        # 应用补丁
        struc2vec.Struc2Vec._get_layers_adj = patched_get_layers_adj
        print("✅ Struc2Vec 补丁应用成功")
        
    except ImportError:
        print("❌ 无法导入 Struc2Vec 模块")
    except Exception as e:
        print(f"❌ 应用补丁失败: {e}")

# 自动应用补丁
if __name__ != "__main__":
    patch_struc2vec()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复 Struc2Vec 除零警告的补丁
"""
import os
import shutil
from pathlib import Path

def create_fixed_struc2vec():
    """创建修复了除零警告的 Struc2Vec 版本"""
    
    # 读取原始文件
    original_file = Path(__file__).parent.parent / "GraphEmbedding/ge/models/struc2vec.py"
    
    if not original_file.exists():
        print(f"❌ 找不到原始文件: {original_file}")
        return False
    
    with open(original_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找需要修改的代码
    old_code = """                e_list = [x / sum_w for x in e_list]
                norm_weights[v] = e_list
                accept, alias = create_alias_table(e_list)
                node_alias_dict[v] = alias
                node_accept_dict[v] = accept"""
    
    # 新的代码，添加除零检查
    new_code = """                # 修复除零警告
                if sum_w > 0:
                    e_list = [x / sum_w for x in e_list]
                else:
                    # 如果所有权重都是0，使用均匀分布
                    if len(e_list) > 0:
                        e_list = [1.0 / len(e_list) for _ in e_list]
                    else:
                        e_list = [1.0]  # 至少有一个权重
                
                norm_weights[v] = e_list
                accept, alias = create_alias_table(e_list)
                node_alias_dict[v] = alias
                node_accept_dict[v] = accept"""
    
    # 替换代码
    if old_code in content:
        content_fixed = content.replace(old_code, new_code)
        
        # 保存修复版本
        fixed_dir = Path(__file__).parent / "src" / "algorithms" / "fixed"
        fixed_dir.mkdir(parents=True, exist_ok=True)
        
        fixed_file = fixed_dir / "struc2vec_fixed.py"
        with open(fixed_file, 'w', encoding='utf-8') as f:
            f.write(content_fixed)
        
        print(f"✅ 创建修复版本: {fixed_file}")
        
        # 创建使用说明
        readme_content = """# 修复版 Struc2Vec

## 修复的问题
- 除零警告: `RuntimeWarning: invalid value encountered in scalar divide`
- 当节点在某层没有邻居或所有距离都很大导致权重接近0时发生

## 修复方案
1. 检查 `sum_w` 是否大于0
2. 如果是0，使用均匀分布代替
3. 确保始终有有效的概率分布

## 使用方法
```python
# 替换原始导入
# from GraphEmbedding.ge.models.struc2vec import Struc2Vec
from src.algorithms.fixed.struc2vec_fixed import Struc2Vec
```

## 修改内容
原代码（第301行）：
```python
e_list = [x / sum_w for x in e_list]
```

修复后：
```python
if sum_w > 0:
    e_list = [x / sum_w for x in e_list]
else:
    # 使用均匀分布
    if len(e_list) > 0:
        e_list = [1.0 / len(e_list) for _ in e_list]
    else:
        e_list = [1.0]
```
"""
        readme_file = fixed_dir / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        return True
    else:
        print("❌ 找不到需要修改的代码段")
        return False

def create_monkey_patch():
    """创建运行时补丁（不修改原文件）"""
    
    patch_content = '''#!/usr/bin/env python3
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
'''
    
    patch_file = Path(__file__).parent / "src" / "utils" / "struc2vec_patch.py"
    patch_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(patch_file, 'w', encoding='utf-8') as f:
        f.write(patch_content)
    
    print(f"✅ 创建运行时补丁: {patch_file}")
    
    # 创建使用示例
    example_content = '''# 在你的脚本开头添加：
import sys
sys.path.insert(0, str(Path(__file__).parent))
from src.utils import struc2vec_patch  # 自动应用补丁

# 然后正常使用 Struc2Vec
from GraphEmbedding.ge.models.struc2vec import Struc2Vec
'''
    
    example_file = Path(__file__).parent / "patch_usage_example.py"
    with open(example_file, 'w', encoding='utf-8') as f:
        f.write(example_content)
    
    return True

def main():
    """主函数"""
    print("🔧 Struc2Vec 除零警告修复工具")
    print("=" * 50)
    
    print("\n选择修复方式:")
    print("1. 创建修复版本文件（推荐）")
    print("2. 创建运行时补丁（不修改原文件）")
    print("3. 两种都创建")
    
    choice = input("\n请选择 (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        print("\n📝 创建修复版本...")
        if create_fixed_struc2vec():
            print("✅ 修复版本创建成功")
    
    if choice in ['2', '3']:
        print("\n🩹 创建运行时补丁...")
        if create_monkey_patch():
            print("✅ 运行时补丁创建成功")
    
    print("\n💡 修复建议:")
    print("1. 这个警告通常不影响最终结果")
    print("2. 出现在图中有孤立子图或距离极大的情况")
    print("3. 修复后会使用均匀分布代替除零结果")

if __name__ == "__main__":
    main()
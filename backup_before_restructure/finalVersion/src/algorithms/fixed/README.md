# 修复版 Struc2Vec

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

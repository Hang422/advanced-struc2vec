# graphlet/algorithm/get_graphlet.py
import json
import os
import subprocess
import numpy as np
import time

def _find_orca_executable():
    project_root = os.path.abspath(
        os.path.join(__file__, os.pardir, os.pardir, os.pardir)
    )
    # 尝试 orca 和 orca.exe
    for name in ("orca", "orca.exe"):
        p = os.path.join(project_root, "orca", name)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    # 再试 PATH
    from shutil import which
    for name in ("orca", "orca.exe"):
        p = which(name)
        if p:
            return p
    raise FileNotFoundError("Cannot find orca binary")

def compute_node_gdv(
    edgelist_path: str,
    k: int = 5,
    orca_bin: str = None,
    mode: str = 'node',
) -> dict[int, np.ndarray]:
    if orca_bin is None:
        orca_bin = _find_orca_executable()

    out_path =  f'{edgelist_path}_GDV.out'
    cmd = [orca_bin, mode, str(k), edgelist_path, out_path]
    # 传给 subprocess.run 的必须是 edgelist_path 的绝对路径
    subprocess.run(cmd, check=True)

    gdv = {}
    with open(out_path, 'r') as f:
        for idx, line in enumerate(f):
            gdv[idx] = np.fromstring(line, dtype=int, sep=' ')
    return gdv



import networkx as nx
import numpy as np
from collections import deque

def graphlet_correlation(M: np.ndarray) -> np.ndarray:
    """
    自定义相关性矩阵：对零方差列输出全 0，避免 NaN / RuntimeWarning。
    M.shape = (n_samples, n_features)
    """
    n, z = M.shape
    X = M - M.mean(axis=0, keepdims=True)
    if n > 1:
        cov = (X.T @ X) / (n - 1)
    else:
        cov = np.zeros((z, z), dtype=float)

    diag = np.diag(cov)
    std = np.sqrt(np.maximum(diag, 0))
    std_safe = std.copy()
    std_safe[std_safe == 0] = 1.0

    C = cov / std_safe[:, None] / std_safe[None, :]
    zero_std = (std == 0)
    if zero_std.any():
        C[zero_std, :] = 0.0
        C[:, zero_std] = 0.0
    return C

def compute_graphlet_dgcd(graph: nx.Graph,
                          node_gdv: np.ndarray,
                          max_layer: int = 5):
    """
    Compute Graphlet-Enhanced structural distances (DGCD) for all node pairs.

    Parameters
    ----------
    graph : networkx.Graph
    node_gdv : np.ndarray, shape (n_nodes, n_orbits)
    max_layer : int or None
        BFS 最大半径；None 时用图的直径。

    Returns
    -------
    distances : dict
        (u, v) -> { layer: cumulative_distance }。
    """
    # 1) 确定 max_layer
    if max_layer is None:
        G_undir = graph.to_undirected()
        if not nx.is_connected(G_undir):
            max_layer = max(
                nx.diameter(G_undir.subgraph(c))
                for c in nx.connected_components(G_undir)
            )
        else:
            max_layer = nx.diameter(G_undir)

    # 2) 节点列表（按升序，以匹配 GDV 文件里“0..n-1”）
    nodes = sorted(graph.nodes())
    n = len(nodes)
    assert node_gdv.shape[0] == n, f"GDV 行数 {node_gdv.shape[0]} ≠ 图节点数 {n}"
    idx_map = {node: i for i, node in enumerate(nodes)}

    # 3) 计算每个节点每层的 DGCM（相关性矩阵）
    z = node_gdv.shape[1]
    layer_dgcm = [dict() for _ in range(n)]
    for ui, u in enumerate(nodes):
        visited = {u}
        queue = deque([(u, 0)])
        layer_nodes = {0: [u]}
        while queue:
            v, d = queue.popleft()
            if d >= max_layer:
                continue
            for w in graph[v]:
                if w not in visited:
                    visited.add(w)
                    queue.append((w, d + 1))
                    layer_nodes.setdefault(d + 1, []).append(w)
        for ℓ in range(max_layer + 1):
            members = layer_nodes.get(ℓ, [])
            if len(members) < 2:
                C = np.zeros((z, z), dtype=float)
            else:
                M = node_gdv[[idx_map[v] for v in members], :]
                C = graphlet_correlation(M)
            layer_dgcm[ui][ℓ] = C

    # 4) 计算两两节点在各层的累积 DGCD
    distances = {}
    for i in range(n):
        for j in range(i + 1, n):
            cum = 0.0
            per_layer = {}
            for ℓ in range(max_layer + 1):
                Ci = layer_dgcm[i][ℓ]
                Cj = layer_dgcm[j][ℓ]
                dℓ = np.linalg.norm(Ci - Cj, ord='fro')
                cum += dℓ
                per_layer[ℓ] = cum
            distances[(nodes[i], nodes[j])] = per_layer

    # out = {f"{u}|{v}": layers for (u, v), layers in distances.items()}
    # with open('dgcd_distances.json', 'w') as fp:
    #     json.dump(out, fp, indent=2, ensure_ascii=False)
    return distances


def compute_graphlet_dgcd_simple(graph: nx.Graph,
                                 node_gdv: np.ndarray,
                                 max_layer: int = None):
    """
    和原版不同的地方在于：每层我们把 DGCM 矩阵所有元素求和，然后
    用两个节点在该层的和的绝对差值作为该层的距离，累加得到最终
    的分层距离字典。
    """
    # 1) 确定 max_layer
    if max_layer is None:
        G0 = graph.to_undirected()
        if not nx.is_connected(G0):
            max_layer = max(
                nx.diameter(G0.subgraph(c))
                for c in nx.connected_components(G0)
            )
        else:
            max_layer = nx.diameter(G0)

    # 2) 节点排序 & 索引映射
    nodes = sorted(graph.nodes())
    n = len(nodes)
    assert node_gdv.shape[0] == n, "GDV 行数必须等于节点数"
    idx = {v: i for i, v in enumerate(nodes)}
    z = node_gdv.shape[1]

    # 3) 为每个节点、每层做 DGCM，然后把矩阵“预累加”成一个标量 sumC[ui][ℓ]
    sumC = [dict() for _ in range(n)]
    for ui, u in enumerate(nodes):
        visited = {u}
        q = deque([(u, 0)])
        layers = {0: [u]}
        while q:
            v, d = q.popleft()
            if d >= max_layer:
                continue
            for w in graph[v]:
                if w not in visited:
                    visited.add(w)
                    q.append((w, d+1))
                    layers.setdefault(d+1, []).append(w)

        for ℓ in range(max_layer+1):
            memb = layers.get(ℓ, [])
            if len(memb) < 2:
                sumC[ui][ℓ] = 0.0
            else:
                M = node_gdv[[idx[w] for w in memb], :]
                C = graphlet_correlation(M)
                sumC[ui][ℓ] = float(C.sum())

    # 4) 两两节点在各层累加绝对差
    distances = {}
    for i in range(n):
        for j in range(i+1, n):
            cum = 0.0
            per = {}
            for ℓ in range(max_layer+1):
                dℓ = abs(sumC[i][ℓ] - sumC[j][ℓ])
                cum += dℓ
                per[ℓ] = cum
            distances[(nodes[i], nodes[j])] = per

    # 5) （可选）保存到 json
    serial = {f"{u}|{v}": layers for (u, v), layers in distances.items()}
    with open('dgcd_simple.json', 'w', encoding='utf-8') as fp:
        json.dump(serial, fp, indent=2, ensure_ascii=False)

    return distances


import json
import networkx as nx
import numpy as np
from collections import deque
from typing import Dict, Tuple
# 下面假设 graphlet_correlation 已经定义好

# 只保留的 orbit 索引
ORBIT_13 = list(range(40))

def compute_graphlet_dgcd_simple13(
    graph: nx.Graph,
    node_gdv: np.ndarray,
    max_layer: int = None
) -> Dict[Tuple[int,int], Dict[int,float]]:
    """
    简化版 DGCD：每层只用前 13 个 orbit，对应的子矩阵求和，
    用两个节点在该层的和的绝对差值作为该层的距离，累加得到 per-layer 距离。
    """
    # 1) 确定 max_layer
    if max_layer is None:
        G0 = graph.to_undirected()
        if not nx.is_connected(G0):
            max_layer = max(nx.diameter(G0.subgraph(c))
                            for c in nx.connected_components(G0))
        else:
            max_layer = nx.diameter(G0)

    # 2) 节点排序 & 索引映射
    nodes = sorted(graph.nodes())
    n = len(nodes)
    assert node_gdv.shape[0] == n, "GDV 行数必须等于节点数"
    idx = {v: i for i, v in enumerate(nodes)}

    # 3) 对每个节点、每层计算 “前13×13 子矩阵的元素和”
    sumC = [dict() for _ in range(n)]
    for ui, u in enumerate(nodes):
        visited = {u}
        q = deque([(u, 0)])
        layers = {0: [u]}
        while q:
            v, d = q.popleft()
            if d >= max_layer:
                continue
            for w in graph[v]:
                if w not in visited:
                    visited.add(w)
                    q.append((w, d+1))
                    layers.setdefault(d+1, []).append(w)

        for ℓ in range(max_layer+1):
            memb = layers.get(ℓ, [])
            if len(memb) < 2:
                sumC[ui][ℓ] = 0.0
            else:
                M = node_gdv[[idx[w] for w in memb], :]
                C = graphlet_correlation(M)
                # 只取前13个 orbit 对应的子矩阵再求和
                C13 = C[np.ix_(ORBIT_13, ORBIT_13)]
                sumC[ui][ℓ] = float(C13.sum())

    # 4) 两两节点按层累加绝对差
    distances = {}
    for i in range(n):
        for j in range(i+1, n):
            cum = 0.0
            per = {}
            for ℓ in range(max_layer+1):
                dℓ = abs(sumC[i][ℓ] - sumC[j][ℓ])
                cum += dℓ
                per[ℓ] = cum
            distances[(nodes[i], nodes[j])] = per

    # 5) （可选）保存到 json
    serial = {f"{u}|{v}": layers for (u, v), layers in distances.items()}
    with open('dgcd_simple13.json', 'w', encoding='utf-8') as fp:
        json.dump(serial, fp, indent=2, ensure_ascii=False)

    return distances


import pandas as pd
# if __name__ == "__main__":
#     start = time.time()
#     G = nx.read_edgelist("brazil-airports.edgelist", nodetype=int)
#     gdv = np.loadtxt("brazil-airports.in_GDV.out")
#     dists = compute_graphlet_dgcd(G, gdv)
#     pd.to_pickle(dists,'b_structural_dist.pkl')
#     end = time.time()
#     print(fw"代码运行时间：{end - start:.6f} 秒")

if __name__ == '__main__':
    import pickle
    with open("b_structural_dist.pkl", "rb") as f:
        loaded_data = pickle.load(f)

    print(loaded_data.keys())

# if __name__ == "__main__":
#     # 1) 定位到项目根
#     start = time.time()
#     project_root = os.path.abspath(
#         os.path.join(__file__, os.pardir, os.pardir, os.pardir)
#     )
#     # 2) 用绝对路径指向 orca/graph.in
#     # graph_in = os.path.join(project_root, "orca", "graph.in")
#     graph_in = 'brazil-airports.in'
#     # 3) 调用
#     gdv = compute_node_gdv(graph_in, k=5)
#     end = time.time()
#     print(f"代码运行时间：{end - start:.6f} 秒")


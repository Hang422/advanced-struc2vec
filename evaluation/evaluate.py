import numpy as np

from GraphEmbedding.ge.classify import read_node_label, Classifier

import sys
sys.path.append('..')
from algorithms.traditional.struc2vec import Struc2Vec

from GraphEmbedding.ge.models.test import test
from GraphEmbedding.ge.models.struc2vec4graphlet import Struc2Vec4graphlet

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

import networkx as nx

from sklearn.manifold import TSNE

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def evaluate_embeddings(embeddings, filename):
    X, Y = read_node_label(filename, skip_head=True)

    tr_frac = 0.8

    print("Training classifier using {:.2f}% nodes...".format(tr_frac * 100))

    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())

    clf.split_train_evaluate(X, Y, tr_frac)


def plot_embeddings(embeddings, ):
    X, Y = read_node_label('../data/flight/labels-brazil-airports.txt', skip_head=True)

    emb_list = []

    for k in X:
        emb_list.append(embeddings[k])

    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)

    node_pos = model.fit_transform(emb_list)

    color_idx = {}

    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])

        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)  # c=node_colors)

    plt.legend()

    plt.show()


import pickle


def fuse_structural_distances(path_a, path_b, output_path, alpha=0.5):
    """
    融合两个结构距离 .pkl 文件，输出加权融合后的新结构距离。

    参数：
    - path_a: 第一个结构距离文件路径（如 graphlet-enhanced）
    - path_b: 第二个结构距离文件路径（如原始）
    - output_path: 融合后保存的路径
    - alpha: 权重系数，fused = alpha * A + (1 - alpha) * B
    """
    with open(path_a, 'rb') as fa, open(path_b, 'rb') as fb:
        dist_a = pickle.load(fa)
        dist_b = pickle.load(fb)

    fused = {}
    keys = set(dist_a.keys()).intersection(dist_b.keys())
    for pair in keys:
        layers_a = dist_a[pair]
        layers_b = dist_b[pair]
        fused[pair] = {}
        shared_layers = set(layers_a.keys()).intersection(layers_b.keys())
        for l in shared_layers:
            fused[pair][l] = alpha * layers_a[l] + (1 - alpha) * layers_b[l]

    with open(output_path, 'wb') as f:
        pickle.dump(fused, f)

    print(f"✅ 融合完成，保存至 {output_path}")


import pickle


def fuse_structural_distances_min(path_a, path_b, output_path):
    """
    融合两个结构距离，按每层最小值融合。
    """
    with open(path_a, 'rb') as fa, open(path_b, 'rb') as fb:
        dist_a = pickle.load(fa)
        dist_b = pickle.load(fb)

    fused = {}
    common_pairs = set(dist_a.keys()).intersection(dist_b.keys())
    for pair in common_pairs:
        layers_a = dist_a[pair]
        layers_b = dist_b[pair]
        fused[pair] = {}
        for l in set(layers_a.keys()).intersection(layers_b.keys()):
            fused[pair][l] = min(layers_a[l], layers_b[l])

    with open(output_path, 'wb') as f:
        pickle.dump(fused, f)

    print(f"✅ 最小值融合完成，保存至 {output_path}")


def fuse_structural_distances_max(path_a, path_b, output_path):
    """
    融合两个结构距离，按每层最大值融合。
    """
    with open(path_a, 'rb') as fa, open(path_b, 'rb') as fb:
        dist_a = pickle.load(fa)
        dist_b = pickle.load(fb)

    fused = {}
    common_pairs = set(dist_a.keys()).intersection(dist_b.keys())
    for pair in common_pairs:
        layers_a = dist_a[pair]
        layers_b = dist_b[pair]
        fused[pair] = {}
        for l in set(layers_a.keys()).intersection(layers_b.keys()):
            fused[pair][l] = max(layers_a[l], layers_b[l])

    with open(output_path, 'wb') as f:
        pickle.dump(fused, f)

    print(f"✅ 最大值融合完成，保存至 {output_path}")


if __name__ == "__main__":
    G = nx.read_edgelist('../data/flight/brazil-airports.edgelist', create_using=nx.DiGraph(), nodetype=None,
                         data=[('weight', int)])

    model = Struc2Vec(G, 10, 80, workers=4, verbose=10, opt1_reduce_len=True, opt2_reduce_sim_calc=True,structural_dist_file="output/structural_dist_fused_max.pkl")
    model.train()
    embeddings = model.get_embeddings()
    evaluate_embeddings(embeddings, "../data/flight/labels-brazil-airports.txt")

    model = Struc2Vec(G, 10, 80, workers=4, verbose=10, opt1_reduce_len=True, opt2_reduce_sim_calc=True,
                      structural_dist_file="output/structural_dist_fused_min.pkl")
    model.train()
    embeddings = model.get_embeddings()
    evaluate_embeddings(embeddings, "../data/flight/labels-brazil-airports.txt")

    # fuse_structural_distances(
    #     path_a="output/structural_dist_brazil-airports.pkl",
    #     path_b="output/structural_dist.pkl",
    #     output_path="output/structural_dist_fused_weighted.pkl",
    #     alpha=0.6
    # )
    # fuse_structural_distances_min(
    #     "output/structural_dist.pkl",
    #     "output/structural_dist_brazil-airports.pkl",
    #     "output/structural_dist_fused_min.pkl"
    # )
    # fuse_structural_distances_max(
    #     "output/structural_dist.pkl",
    #     "output/structural_dist_brazil-airports.pkl",
    #     "output/structural_dist_fused_max.pkl"
    # )

    # model = Struc2Vec(G, 10, 80, workers=4, verbose=10, opt1_reduce_len=True, opt2_reduce_sim_calc=True,
    #                   structural_dist_file="output/structural_dist_fused_weighted.pkl")
    # model.train()
    # embeddings = model.get_embeddings()
    # evaluate_embeddings(embeddings, "../data/lastfm_asia/lastfm_asia_labels.txt")
    #
    # model = Struc2Vec(G, 10, 80, workers=4, verbose=10, opt1_reduce_len=True, opt2_reduce_sim_calc=True)
    # model.train()
    # embeddings = model.get_embeddings()
    # evaluate_embeddings(embeddings, "../data/flight/labels-brazil-airports.txt")

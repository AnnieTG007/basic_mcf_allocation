"""由 main 导入调用，读取拓扑 JSON，生成直连距离矩阵和按总边长排序的候选路径。
输入边长使用 km，节点编号从零连续递增，返回 NetworkX 图、数组或路径列表。"""
import json
from itertools import islice
from pathlib import Path

import networkx as nx
import numpy as np


def validate_graph(graph):
    """检查无向、无重边、连续整数节点以及有限正边长；允许不连通，不允许自环。"""
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("Expected an undirected simple graph")
    nodes = list(graph.nodes)
    if (not nodes or any(type(n) is not int for n in nodes)
            or sorted(nodes) != list(range(len(nodes)))):
        raise ValueError("Nodes must be consecutive integers starting at zero")
    for u, v, data in graph.edges(data=True):
        length = data.get("length_km")
        if (u == v or isinstance(length, bool)
                or not isinstance(length, (int, float))
                or not np.isfinite(length) or length <= 0):
            raise ValueError("Each edge needs a finite positive length_km and distinct endpoints")


def load_topology(path):
    """从 path 指定的 JSON 加载无向拓扑，拒绝重复边和未声明的节点。"""
    # JSON 需含 directed=false、完整 nodes 列表及 edges；每边如 {source: 0, target: 1, length_km: 10}。
    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream)
    if data.get("directed") is not False:
        raise ValueError("JSON must explicitly specify directed=false")
    nodes = data["nodes"]
    if any(type(n) is not int for n in nodes) or len(set(nodes)) != len(nodes):
        raise ValueError("Node IDs must be unique integers")
    graph = nx.Graph(name=data.get("name", Path(path).stem))
    # 加载时固定节点和边的插入顺序，为等长候选提供可重复的输入。
    graph.add_nodes_from(sorted(nodes))
    for edge in sorted(data["edges"], key=lambda e: (e["source"], e["target"])):
        u, v = edge["source"], edge["target"]
        if type(u) is not int or type(v) is not int or u not in graph or v not in graph:
            raise ValueError("Edge endpoint is not a declared integer node")
        if graph.has_edge(u, v):
            raise ValueError(f"Duplicate undirected edge: {u}, {v}")
        graph.add_edge(u, v, length_km=edge["length_km"])
    validate_graph(graph)
    return graph


def distance_matrix(graph, unit="km"):
    """返回直连距离矩阵：对角线为零，无边为 inf；并非最短路距离矩阵。"""
    validate_graph(graph)
    if unit not in ("km", "m"):
        raise ValueError("unit must be 'km' or 'm'")
    # unit 仅接受 km/m；例如 unit="m" 将文件中的 10 km 换为 10000 m。
    matrix = nx.to_numpy_array(graph, nodelist=range(len(graph)),
                               weight="length_km", nonedge=np.inf)
    np.fill_diagonal(matrix, 0.0)
    return matrix * (1000.0 if unit == "m" else 1.0)


def k_shortest_paths(graph, source, target, k):
    """按总边长返回至多 k 条简单路径及距离（km），不连通时返回空列表。"""
    # source/target 是节点编号，如 0/1；k 为路径数量上限，如 3。
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    if source not in graph or target not in graph:
        raise nx.NodeNotFound("Source or target is not in the graph")
    if source == target:
        return [([source], 0.0)]
    # 等长路径沿用 NetworkX 的生成顺序；任意外部图的插入顺序由调用方保证。
    # 单路径与多路径分支没有额外统一的同分排序，不承诺复现旧 Kshort 顺序。
    try:
        if k == 1:
            cost, path = nx.single_source_dijkstra(graph, source, target, weight="length_km")
            return [(path, float(cost))]
        paths = islice(nx.shortest_simple_paths(graph, source, target,
                                                weight="length_km"), k)
        return [(path, float(nx.path_weight(graph, path, "length_km"))) for path in paths]
    except nx.NetworkXNoPath:
        return []

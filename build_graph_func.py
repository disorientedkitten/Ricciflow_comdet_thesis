import networkx as nx
import numpy as np
import math
import ast
from sklearn import preprocessing
from sklearn import metrics
import pandas as pd
import random
from collections import Counter
import GraphRicciCurvature
from networkx.drawing.layout import spring_layout
import csv
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

import logging

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.ERROR)

from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci


def remove_degree_zero_nodes_efficient(graph):
    """
    Удаляет из графа вершины со степенью 0 (изолированные вершины).

    Args:
        graph (nx.Graph): Граф, представленный объектом NetworkX.

    Returns:
        nx.Graph: Новый граф, в котором отсутствуют вершины со степенью 0.
    """
    return graph.subgraph([node for node, degree in graph.degree() if degree > 0])


def kruskal_mst(edges, num_vertices):
    """
    Находит минимальное остовное дерево (MST) в графе, используя алгоритм Крускала.

    Args:
        edges (list[tuple]): Список рёбер графа в формате (u, v, weight).
        num_vertices (int): Количество вершин в графе.

    Returns:
        tuple:
            - mst (list[tuple]): Список рёбер MST в формате (u, v, weight).
            - max_weight_in_mst (int/float): Максимальный вес ребра в MST.
    """
    edges.sort(key=lambda x: x[2])
    uf = UnionFind(num_vertices)
    mst = []
    max_weight_in_mst = 0
    for u, v, weight in edges:
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            mst.append((u, v, weight))
            max_weight_in_mst = weight
    return mst, max_weight_in_mst


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1


def find_max_weight_in_mst(file_path, indices):
    """
    Считывает матрицу расстояний из файла, строит граф по заданным индексам и находит
    максимальный вес ребра в минимальном остовном дереве (MST) этого графа.

    Args:
        file_path (str): Путь к файлу с матрицей расстояний (формат CSV).
        indices (list): Список индексов строк/столбцов матрицы для построения графа.

    Returns:
        float: Максимальный вес ребра в MST построенного графа.

    Raises:
        FileNotFoundError: Если файл по указанному пути не существует.
        ValueError: Если матрица расстояний имеет некорректный формат.
    """
    edges = []
    try:
        with open(file_path, mode="r") as file:
            for i, line in enumerate(file):
                row = line.strip().split(",")
                matrix_dist = np.array([float(x) for x in row])
                for j in range(len(indices)):
                    if i < j:
                        edges.append((i, j, matrix_dist[j]))
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл {file_path} не найден")
    except (ValueError, IndexError):
        raise ValueError("Некорректный формат матрицы расстояний")

    num_vertices = len(indices)
    mst, max_weight = kruskal_mst(edges, num_vertices)
    return max_weight


def find_treshhold(filename, indices, rule="avg", percentile=50):
    """
    Вычисляет treshhold для построения графа.
    Args:
        indices: лист индексов вершин графа.
        rule: правило выбора treshhold
        percentile: параметр для rule="percentile"
    Returns:
        treshhold

    """
    all_distance = []
    with open(filename, "r") as file:
        for i, line in enumerate(file):
            row = line.strip().split(",")
            matrix_dist = np.array([float(x) for x in row])
            for j in range(len(indices)):
                if i < j:
                    all_distance.append(matrix_dist[j])
    if rule == "avg":
        return np.mean(all_distance)
    if rule == "median":
        return np.mediane(all_distance)
    if rule == "percentile":
        return np.percentile(all_distance, percentile)
    if rule == "connectivity":
        return find_max_weight_in_mst(filename, indices)


def get_indices(x_train, y_train, l_bound_target=8 / 100, u_bound_target=12 / 100, sample_size=100):
    """
    Выбирает подвыборку индексов из обучающих данных, обеспечивая баланс классов в заданных границах.

    Args:
        x_train (list/array): Обучающие данные (признаки).
        y_train (list/array): Метки классов, соответствующие x_train.
        l_bound_target (float, optional): Нижняя граница допустимой доли каждого класса (по умолчанию 8%).
        u_bound_target (float, optional): Верхняя граница допустимой доли каждого класса (по умолчанию 12%).
        sample_size (int, optional): Размер требуемой подвыборки (по умолчанию 100).

    Returns:
        list: Список индексов, удовлетворяющих условиям баланса классов.

    Raises:
        AssertionError: Если sample_size превышает размер данных (len(x_train)).

    Примечание:
        - Функция выводит распределение классов в финальной выборке.
    """
    assert sample_size <= len(x_train), "sample_size не может быть больше, чем размер набора данных"

    indices = random.sample(range(len(x_train)), sample_size)
    x_sample = [x_train[i] for i in indices]
    y_sample = [y_train[i] for i in indices]
    counts = Counter(y_sample)

    while max(counts.values()) >= u_bound_target * sample_size or min(counts.values()) <= l_bound_target * sample_size:
        indices = random.sample(range(len(x_train)), sample_size)
        y_sample = [y_train[i] for i in indices]
        counts = Counter(y_sample)

    print(counts)
    return indices


def top_x_labels(y_train: list, x: int) -> set:
    """Возвращает множество X наиболее частых меток в наборе данных.

    Args:
        y_train (list): Список меток классов.
        x (int): Количество наиболее частых меток для возврата.

    Returns:
        set: Множество X самых распространённых меток.

    Raises:
        ValueError: Если x превышает количество уникальных меток.
    """
    label_counts = Counter(y_train)
    if x > len(label_counts):
        raise ValueError("x не может превышать количество уникальных меток")
    return {label for label, _ in label_counts.most_common(x)}


def get_indices_from_top_label(x_train: list, y_train: list, x: int, max_samples: int = 100) -> list:
    """Выбирает индексы данных для X наиболее частых классов с ограничением выборки.

    Args:
        x_train (list): Массив признаков обучающей выборки.
        y_train (list): Массив меток классов.
        x (int): Количество топовых классов для выборки.
        max_samples (int, optional): Максимальный размер выборки. По умолчанию 100.

    Returns:
        list: Список индексов, соответствующих выбранным классам.

    Examples:
        >>> indices = get_indices(X_train, y_train, 5)
        >>> len(indices) <= 3002
        True
    """
    target_labels = top_x_labels(y_train, x)
    indices = [i for i, label in enumerate(y_train) if label in target_labels][:max_samples]

    print(f"Распределение классов в выборке: {Counter(y_train[i] for i in indices)}")
    return indices


def get_random_sample(filename, x_train, y_train, indices, threshold):
    """
    Берет sample_size случайных точек из MNIST и строит по ним граф метрический граф.
    Точки из MNIST выбираются так чтобы каждого таргета было не меньше l_bound_target u_bound_target.
    Выводится количество каждого таргета.
    """
    if len(x_train) != len(y_train):
        raise ValueError("x_train и y_train должны иметь одинаковую длину.")
    x_sample = [x_train[i] for i in indices]
    y_sample = [y_train[i] for i in indices]
    graph_MNIST = nx.Graph()
    for i in range(len(x_sample)):
        node = str(x_sample[i].flatten().tolist())
        label = y_sample[i]
        graph_MNIST.add_nodes_from([(node, {"label": label})])
    print("threshold = ", threshold)
    with open(filename, "r") as file:
        for i, line in enumerate(file):
            row = line.strip().split(",")
            matrix_dist = np.array([float(x) for x in row])
            for j in range(len(indices)):
                if i < j:
                    if matrix_dist[j] <= threshold:
                        graph_MNIST.add_edge(
                            str(x_train[indices[i]].flatten().tolist()),
                            str(x_train[indices[j]].flatten().tolist()),
                            weight=matrix_dist[j],
                        )
    return np.array(x_sample), np.array(y_sample), graph_MNIST


def table_by_graph(df, graph):
    """
    Args:
        graph: Граф NetworkX.
    Returns:
        Строки MNIST соответствующие вершинам графа.
    """
    x_graph = []
    for item_str in graph.nodes:
        try:
            item_list = ast.literal_eval(item_str)
            item_array = np.array(item_list)  # Предполагаем одномерный массив
            x_graph.append(item_array)
        except (ValueError, SyntaxError, TypeError) as e:
            print(f"Ошибка при обработке строки '{item_str}': {e}")
    len_embed = len(x_graph[0])
    x_graph = pd.DataFrame(x_graph)
    on = [x for x in range(len_embed)]
    inter = pd.merge(x_graph, df, how="inner", on=on)
    return x_graph, inter

import networkx as nx
import numpy as np
import pandas as pd
import GraphRicciCurvature
from networkx.drawing.layout import spring_layout
import csv
import matplotlib.pyplot as plt

import logging

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.ERROR)

from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci

def two_counts_gist(dict1, dict2):
    """
    Строит сравнительную гистограмму значений двух словарей.

    Args:
        dict1 (dict): Первый словарь с данными (исходный граф).
        dict2 (dict): Второй словарь с данными (модифицированный граф).

    Returns:
        None: Отображает matplotlib гистограмму.
    """
    keys = sorted(list(set(dict1.keys()).union(dict2.keys())))
    values1 = [dict1.get(key, 0) for key in keys]
    values2 = [dict2.get(key, 0) for key in keys]

    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(keys, values1, width, label="изначальный граф")
    rects2 = ax.bar([x + width for x in keys], values2, width, label="граф после изменений")

    ax.set_ylabel("количества")
    ax.set_xlabel("таргеты")
    ax.set_title("гистограмма")
    ax.set_xticks([x + width / 2 for x in keys])
    ax.set_xticklabels(keys)
    ax.legend()

    fig.tight_layout()
    plt.show()


def two_visualisation_target_colors(graph1, graph2, iteration_spring=50, legend=False):
    """
    Визуализирует два графа с цветовой кодировкой по меткам узлов.

    Args:
        graph1 (nx.Graph): Первый граф для визуализации.
        graph2 (nx.Graph): Второй граф для визуализации.
        iteration_spring (int, optional): Количество итераций для spring layout.
        legend (bool, optional): Флаг отображения легенды.

    Returns:
        None: Отображает matplotlib графики.
    """
    pos1 = nx.spring_layout(graph1, iterations=iteration_spring)
    pos2 = nx.spring_layout(graph2, iterations=iteration_spring)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    all_labels = set()
    all_labels.update(graph1.nodes[node].get("label") for node in graph1.nodes())
    all_labels.update(graph2.nodes[node].get("label") for node in graph2.nodes())
    
    label_to_num = {label: i for i, label in enumerate(sorted(all_labels))}
    
    labels1 = [label_to_num[graph1.nodes[node].get("label")] for node in graph1.nodes()]
    labels2 = [label_to_num[graph2.nodes[node].get("label")] for node in graph2.nodes()]

    nx.draw(
        graph1,
        pos1,
        with_labels=False,
        node_size=15,
        node_color=labels1,
        font_size=10,
        ax=axes[0],
        width=0.3,
        cmap=plt.cm.tab20,
    )
    axes[0].set_title("Graph 1")

    nx.draw(
        graph2,
        pos2,
        with_labels=False,
        node_size=15,
        node_color=labels2,
        font_size=10,
        ax=axes[1],
        width=0.3,
        cmap=plt.cm.tab20,
    )
    axes[1].set_title("Graph 2")

    if legend:
        patches = [plt.Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor=plt.cm.tab20(i / (len(label_to_num)-1)), 
                            markersize=10, label=label) 
                for label, i in label_to_num.items()]
        
        fig.legend(handles=patches, bbox_to_anchor=(1.05, 0.5), loc='center left')
        
    plt.tight_layout()
    plt.show()


def two_visualisation_concomp_colors(graph1, graph2, iteration_spring=5):
    """
    Визуализирует два графа с цветовой кодировкой по компонентам связности.

    Args:
        graph1 (nx.Graph): Первый граф для визуализации.
        graph2 (nx.Graph): Второй граф для визуализации.
        iteration_spring (int, optional): Количество итераций для spring layout.

    Returns:
        None: Отображает matplotlib графики.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    def draw_graph(graph, ax):
        components = list(nx.connected_components(graph))
        colors = ["red", "blue", "green", "yellow", "orange", "purple"]
        pos = nx.spring_layout(graph, iterations=iteration_spring)
        node_size = 10
        for i, component in enumerate(components):
            nx.draw_networkx_nodes(
                graph, pos, nodelist=list(component), node_color=colors[i % len(colors)], node_size=node_size, ax=ax
            )
            nx.draw_networkx_edges(graph, pos, edgelist=graph.edges(list(component)), ax=ax)
        ax.axis("off")

    draw_graph(graph1, axes[0])
    draw_graph(graph2, axes[1])

    plt.tight_layout()
    plt.show()


def show_results(G, curvature="ricciCurvature"):
    """
    Отображает гистограммы кривизн Ricci и весов ребер графа.

    Args:
        G (nx.Graph): Граф для анализа.
        curvature (str, optional): Название атрибута кривизны.

    Returns:
        None: Отображает matplotlib гистограммы.
    """
    plt.subplot(2, 1, 1)
    ricci_curvtures = nx.get_edge_attributes(G, curvature).values()
    plt.hist(ricci_curvtures, bins=20)
    plt.xlabel("Ricci curvature")
    plt.title("Histogram of Ricci Curvatures")

    plt.subplot(2, 1, 2)
    weights = nx.get_edge_attributes(G, "weight").values()
    plt.hist(weights, bins=20)
    plt.xlabel("Edge weight")
    plt.title("Histogram of Edge weights")

    plt.tight_layout()


def graph_main_info(graph):
    """
    Выводит основную информацию о графе.

    Args:
        graph (nx.Graph): Граф для анализа.

    Returns:
        None: Выводит информацию в консоль.
    """
    print(f"Количество вершин в графе: {graph.number_of_nodes()}")
    print(f"Количество ребер в графе: {graph.number_of_edges()}")
    num_components = nx.number_connected_components(graph)
    print(f"Количество компонент связности: {num_components}")
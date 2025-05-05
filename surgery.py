import networkx as nx
import numpy as np
import pandas as pd
from collections import Counter
import GraphRicciCurvature

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.ERROR)

from GraphRicciCurvature.OllivierRicci import OllivierRicci




import networkx as nx
import matplotlib.pyplot as plt
from GraphRicciCurvature.OllivierRicci import OllivierRicci
def simple_surgery(G_origin: nx.Graph, weight: str = "weight", cut: float = 0) -> nx.Graph:
    """Выполняет срезку графа, удаляя ребра с весом выше заданного порога.
    
    Удаляет ребра, значение веса которых превышает заданный порог. Если порог не указан,
    автоматически вычисляет его как 60% от диапазона значений выше 1.0.

    Args:
        G_origin (nx.Graph): Исходный граф для обработки.
        weight (str, optional): Атрибут ребра, используемый как вес. По умолчанию "weight".
        cut (float, optional): Пороговое значение для удаления ребер. Если 0, вычисляется автоматически.

    Returns:
        nx.Graph: Новый граф с удаленными ребрами согласно критерию.

    Raises:
        AssertionError: Если указано отрицательное значение порога.
    """
    G = G_origin.copy()
    edge_weights = nx.get_edge_attributes(G, weight)

    assert cut >= 0, "Значение порога не может быть отрицательным"
    if not cut:
        cut = (max(edge_weights.values()) - 1.0) * 0.6 + 1.0  # Автоматический расчет порога

    edges_to_remove = [(n1, n2) for n1, n2 in G.edges() if G[n1][n2][weight] > cut]
    G.remove_edges_from(edges_to_remove)

    return G


def another_surgery(G_origin: nx.Graph, weight: str = "weight", cut: float = 0) -> nx.Graph:
    """Выполняет срезку графа, удаляя ребра с весом ниже заданного порога.
    
    Удаляет ребра, значение веса которых ниже заданного порога. Если порог не указан,
    автоматически вычисляет его как 60% от диапазона значений выше 1.0.

    Args:
        G_origin (nx.Graph): Исходный граф для обработки.
        weight (str, optional): Атрибут ребра, используемый как вес. По умолчанию "weight".
        cut (float, optional): Пороговое значение для удаления ребер. Если 0, вычисляется автоматически.

    Returns:
        nx.Graph: Новый граф с удаленными ребрами согласно критерию.

    Raises:
        AssertionError: Если указано отрицательное значение порога.
    """
    G = G_origin.copy()
    edge_weights = nx.get_edge_attributes(G, weight)

    assert cut >= 0, "Значение порога не может быть отрицательным"
    if not cut:
        cut = (max(edge_weights.values()) - 1.0) * 0.6 + 1.0  # Автоматический расчет порога

    edges_to_remove = [(n1, n2) for n1, n2 in G.edges() if G[n1][n2][weight] < cut]
    G.remove_edges_from(edges_to_remove)

    return G
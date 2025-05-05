import networkx as nx
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
import pandas as pd
import random
from collections import Counter
import GraphRicciCurvature
import csv
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.ERROR)

from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci

def count_distance(node, second_node, metric):
    """
    Вычисляет расстояние между двумя векторами согласно заданной метрике.

    Args:
        node (np.array): Первый вектор.
        second_node (np.array): Второй вектор.
        metric (str): Метрика расстояния. Допустимые значения:
            - "Euclidian": Евклидово расстояние (L2 норма)
            - "Manhetten": Манхэттенское расстояние (L1 норма)
            - "inf_norm": Бесконечная норма (максимальное абсолютное значение разности)
            - "cosine": Косинусное сходство
            - "mse": Среднеквадратичная ошибка

    Returns:
        float: Расстояние между векторами согласно выбранной метрике.
    """
    if metric == "Euclidian":
        return np.linalg.norm(node - second_node, 2)
    if metric == "Manhetten":
        return np.linalg.norm(node - second_node, 1)
    if metric == "inf_norm":
        return np.linalg.norm(node - second_node, np.inf)
    if metric == "cosine":
        cosine_sim = cosine_similarity(node.flatten().reshape(1, -1), second_node.flatten().reshape(1, -1))
        return cosine_sim[0][0]
    if metric == "mse":
        return np.mean((node - second_node) ** 2)


def calculate_and_save_distances(x_train, indices, filename="mnist_dist_sample.csv", metric="Euclidian"):
    """
    Вычисляет попарные расстояния между элементами выборки и сохраняет в CSV файл.

    Args:
        x_train (np.array): Массив данных, где каждая строка представляет точку данных.
        indices (list): Список индексов строк для вычисления расстояний.
        filename (str, optional): Имя CSV файла для сохранения. По умолчанию "mnist_dist_sample.csv".
        metric (str, optional): Метрика расстояния. По умолчанию "Euclidian".

    Returns:
        None: Создает файл с матрицей расстояний.
    """
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for i in indices:
            row_distances = []
            for j in indices:
                node = x_train[i]
                second_node = x_train[j]
                distance = count_distance(node, second_node, metric)
                row_distances.append(distance)
            writer.writerow(row_distances)


def build_investments_dist(filename, indices):
    """
    Строит гистограмму распределения расстояний из файла с матрицей расстояний.

    Args:
        filename (str): Путь к файлу с матрицей расстояний.
        indices (list): Список индексов, используемых при построении матрицы.

    Returns:
        None: Отображает гистограмму распределения расстояний.
    """
    all_distance = []
    with open(filename, "r") as file:
        for i, line in enumerate(file):
            row = line.strip().split(",")
            matrix_dist = np.array([float(x) for x in row])
            for j in range(len(indices)):
                if i < j:
                    all_distance.append(matrix_dist[j])

    plt.figure(figsize=(10, 6))
    plt.hist(all_distance, bins=100, color="skyblue", edgecolor="black")
    plt.title("Гистограмма частоты данных", fontsize=16)
    plt.xlabel("Значения", fontsize=14)
    plt.ylabel("Частота", fontsize=14)
    plt.grid(axis="y", alpha=0.5)
    plt.show()
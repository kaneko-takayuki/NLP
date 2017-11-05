# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def plot(left, input_file, label):
    """
    グラフ1つ分を描画する
    :param left: 横軸の長さ
    :param input_file: 描画する精度ファイル
    :param label: どの色で描画するか
    :return: なし
    """
    height = []
    with open(input_file, 'r') as f:
        for line in f:
            items = line.split('\t')
            height.append(float(items[1]))
    plt.plot(np.array(left), np.array(height), linewidth=1, label=label)


def draw_graph(left, input_files, labels, xlabel, ylabel, title):
    """
    折れ線グラフを描画する
    :param left: 横軸の長さ
    :param input_files: 描画する精度ファイルリスト
    :param labels: どの色でグラフを表示するのか
    :param xlabel: 横軸の説明
    :param ylabel: 縦軸の説明
    :param title: グラフのタイトル
    :return: なし
    """
    for input_file, label in zip(input_files, labels):
        plot(left, input_file, label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

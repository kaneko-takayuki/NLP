# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def plot(left, input_file, row, label):
    """
    グラフ1つ分を描画する
    :param left: 横軸の長さ
    :param input_file: 描画する精度ファイル
    :param row: ファイルの中のどの列をに対するグラフを描画するか
    :param label: グラフのラベル
    :return: なし
    """
    height = []
    with open(input_file, 'r') as f:
        for line in f:
            items = line.split('\t')
            height.append(float(items[row]))
    plt.plot(np.array(left), np.array(height), linewidth=2, label=label)


def draw_graph_files(left, input_files, row, labels, xlabel, ylabel, title, min_ylim, max_ylim):
    """
    全体正答率の推移を複数のファイルを選択してグラフとして描画する
    :param left: 横軸の長さ
    :param input_files: 描画する精度ファイルリスト
    :param row: 左から何番目の数値をグラフとしてプロットするか
    :param labels: グラフのラベルリスト
    :param xlabel: 横軸の説明
    :param ylabel: 縦軸の説明
    :param title: グラフのタイトル
    :param min_ylim 縦軸の最小値
    :param max_ylim 縦軸の最大値
    :return: なし
    """
    fp = FontProperties(fname='/home/kaneko-takayuki/IPAfont00303/ipag.ttf')  # 全角フォントの指定

    for input_file, label in zip(input_files, labels):
        plot(left, input_file, row, label)
    plt.ylim([min_ylim, max_ylim])
    plt.title(title, fontproperties=fp)
    plt.xlabel(xlabel, fontproperties=fp)
    plt.ylabel(ylabel, fontproperties=fp)
    plt.legend(prop=fp)
    plt.show()


def draw_graph_file(left, input_file, rows, labels, xlabel, ylabel, title, min_ylim, max_ylim):
    """
    全体正答率の推移を複数のファイルを選択してグラフとして描画する
    :param left: 横軸の長さ
    :param input_file: 描画する精度ファイル
    :param rows: 左から何番目の数値をグラフとしてプロットするか(複数指定)
    :param labels: グラフのラベルリスト
    :param xlabel: 横軸の説明
    :param ylabel: 縦軸の説明
    :param title: グラフのタイトル
    :param min_ylim 縦軸の最小値
    :param max_ylim 縦軸の最大値
    :return: なし
    """
    fp = FontProperties(fname='/home/kaneko-takayuki/IPAfont00303/ipag.ttf')  # 全角フォントの指定

    for row, label in zip(rows, labels):
        plot(left, input_file, row, label)
    plt.ylim([min_ylim, max_ylim])
    plt.title(title, fontproperties=fp)
    plt.xlabel(xlabel, fontproperties=fp)
    plt.ylabel(ylabel, fontproperties=fp)
    plt.legend(prop=fp)
    plt.show()

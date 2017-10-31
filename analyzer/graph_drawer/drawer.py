# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def plot(left, input_file, label):
    height = []
    with open(input_file, 'r') as f:
        for line in f:
            items = line.split('\t')
            height.append(float(items[1]))
    plt.plot(np.array(left), np.array(height), linewidth=1, label=label)


def draw_graph(left, input_files, labels):
    for input_file, label in zip(input_files, labels):
        plot(left, input_file, label)
    plt.legend()
    plt.show()

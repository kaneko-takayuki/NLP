# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def plot(left, input_file):
    height = []
    with open(input_file, 'r') as f:
        for line in f:
            items = line.split('\t')
            height.append(float(items[1]))
    plt.plot(np.array(left), np.array(height), linewidth=3)


def draw_graph(left, input_files):
    for input_file in input_files:
        plot(left, input_file)
    plt.show()

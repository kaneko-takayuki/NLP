# -*- coding: utf-8 -*-

"""
amazon_corpusのラベルの個数をカウントするスクリプト
"""

import os
from econvertor import spliter


def main():
    label_n = [0 for _ in range(5)]
    space_n = [0 for _ in range(5)]

    project_path = os.path.dirname(os.path.abspath(__file__)) + "/../"
    with open(project_path + "amazon_corpus/data/books/all.tsv") as f:
        for line in f:
            items = line.split('\t')
            label_n[int(items[0]) - 1] += 1
            space_n[int(items[0]) - 1] += spliter.word_count(items[1])

    print("ラベル1: " + str(label_n[0]) + str(space_n[0]))
    print("ラベル2: " + str(label_n[1]) + str(space_n[1]))
    print("ラベル3: " + str(label_n[2]) + str(space_n[2]))
    print("ラベル4: " + str(label_n[3]) + str(space_n[3]))
    print("ラベル5: " + str(label_n[4]) + str(space_n[4]))


if __name__ == '__main__':
    main()

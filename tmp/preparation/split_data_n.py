# -*- coding: utf-8 -*-
"""
input_fileのデータをn分割して
output_dirに
extensionの拡張子で
出力するスクリプト
"""

import numpy as np


input_file = "/home/kaneko-takayuki/NLP/amazon_corpus/data/jp/books/all.tsv"
output_dir = "/home/kaneko-takayuki/NLP/amazon_corpus/data/jp/books/"
extension = ".tsv"
n = 5

if __name__ == "__main__":
    # 一時的にデータを全てtmp_dataに格納
    tmp_data = []
    with open(input_file, "r") as f:
        for line in f:
            tmp_data.append(line)

    # 5つのデータセットを作り出して出力
    tmp_data = np.array(tmp_data)
    perm = np.random.permutation(len(tmp_data))
    mass = int(len(tmp_data) / 5)
    for i in range(5):
        i_dataset = tmp_data[perm[i*mass:(i*mass)+mass]]
        with open(output_dir+"dataset"+str(i+1)+extension, "w") as o:
            o.write("".join(i_dataset))

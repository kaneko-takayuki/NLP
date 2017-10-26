# -*- coding: utf-8 -*-

import sys
import argparse

import BoWSVMRunner
import W2VSVMRunner


def prepare_data(file_name):
    """
    file_nameを読み込み、リストで返す
    :param file_name: 読み込むtsvファイル
    :return: [[文1, ラベル1], [文2, ラベル2], ..., [文N, ラベルN]]
    """
    data = []
    with open(file_name, "r") as f:
        for line in f:
            items = line.split("\t")
            label = 1 if items[4] == "p" else 0
            sentence = items[5]
            data.append([sentence, label])

    return data


if __name__ == "__main__":
    # パラメータを読み取る
    parser = argparse.ArgumentParser(description='FFNN_parameter')
    parser.add_argument("--windowsize", "-w", type=int, default=3)
    args = parser.parse_args()

    args.windowsize = 7

    # ネットワークインスタンス作成
    net = W2VSVMRunner.W2VSVMRunner()

    # 5分割交差検定
    for k in range(1, 6):
        print(str(k) + "分割目...")
        # 学習データ
        train_data = []
        # テストデータ
        test_data = []
        for j in range(1, 6):
            if k == j:
                test_data.extend(prepare_data("tsukuba_corpus/data/dataset" + str(j) + ".tsv"))
            else:
                train_data.extend(prepare_data("tsukuba_corpus/data/dataset" + str(j) + ".tsv"))

        net.set_train_data(train_data)
        net.set_test_data(test_data)
        net.train(w=args.windowsize)
        net.test(w=args.windowsize, file_name="tsukuba_corpus/SVM_W2V/Window"+str(args.windowsize)+"/out/cross_validation" + str(k) + ".txt")
        print("完了")

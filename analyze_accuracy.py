# -*- coding: utf-8 -*-

"""
精度を分析する
"""

import sys
import Constant


def analyze_accuracy(input_file, output_file):
    max_accuracy = 0.0    # 最大精度
    min_accuracy = 100.0  # 最小精度
    sum_accuracy = 0.0    # 精度の和
    data_n = 0            # データ数

    with open(input_file, "r") as f:
        for line in f:
            accuracy = float(line.strip())  # 精度を取り出す
            # 最大精度・最小精度を更新
            if max_accuracy < accuracy:
                max_accuracy = accuracy
            if min_accuracy > accuracy:
                min_accuracy = accuracy

            # 精度和・データ数を更新
            sum_accuracy += accuracy
            data_n += 1

    with open(output_file, "w") as f:
        f.write("最大精度: " + str(max_accuracy) + "\n")
        f.write("最小精度: " + str(min_accuracy) + "\n")
        f.write("平均精度: " + str(sum_accuracy/data_n) + "\n")


# reference_path内に置いた精度ファイルに関して、最大精度を返す
def output_max_accuracy(reference_path):
    max_accuracy = 0.0
    for k in range(1, 6):
        input_file = reference_path + "cross_validation" + str(k) + "/analyzed_accuracy_file.txt"
        with open(input_file, "r") as f:
            items = f.readline().strip().split(" ")
            accuracy = float(items[1])
            if max_accuracy < accuracy:
                max_accuracy = accuracy
    return max_accuracy

if __name__ == "__main__":
    # FFNN用
    """
    window_sizes = ["Window3", "Window5", "Window7", "MultiWindow35", "MultiWindow37", "MultiWindow57", "MultiWindow357"]

    for w in window_sizes:
        base_path = Constant.BASE_PATH + "tsukuba_corpus/FFNN_W2V/" + w + "/zero/shaped_out/"

        # softmaxについて
        for k in range(1, 6):
            reference_path = base_path + "softmax/cross_validation" + str(k) + "/"
            analyze_accuracy(input_file=reference_path+"accuracy_file.txt",
                         output_file=reference_path+"analyzed_accuracy_file.txt")

        max_accuracy = output_max_accuracy(reference_path=base_path + "softmax/")
        sys.stdout.write(base_path + "softmax: ")
        print(max_accuracy)

        # 多数決について
        for k in range(1, 6):
            reference_path = base_path + "majority/cross_validation" + str(k) + "/"
            analyze_accuracy(input_file=reference_path+"accuracy_file.txt",
                         output_file=reference_path+"analyzed_accuracy_file.txt")

        max_accuracy = output_max_accuracy(reference_path=base_path + "majority/")
        sys.stdout.write(base_path + "majority: ")
        print(max_accuracy)
    """


    # LSTM用
    base_path = Constant.BASE_PATH + "tsukuba_corpus/LSTM_W2V/shaped_out/cross_validation"
    for k in range(1, 6):
        reference_path = base_path + str(k) + "/"
        analyze_accuracy(input_file=reference_path + "accuracy_file.txt",
                        output_file=reference_path + "analyzed_accuracy_file.txt")

        max_accuracy = output_max_accuracy(Constant.BASE_PATH + "tsukuba_corpus/LSTM_W2V/shaped_out/")
        sys.stdout.write("LSTM: ")
        print(max_accuracy)

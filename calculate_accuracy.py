# -*- coding: utf-8 -*-

"""
まとめた結果ファイルを元に精度を計算
"""

import Constant


def calculate_accuracy(input_file):
    """
    ファイルの中身を参照して、精度を計算
    :param input_file: 結果ファイル
    :return: 精度(float型)
    """
    with open(input_file, "r") as f:
        data_n = 0     # 正解数
        correct_n = 0  # データ数

        # 1データずつ参照
        for line in f:
            items = line.strip().split("\t")
            data_n += 1
            if items[0] == items[1]:
                correct_n += 1

    return float(correct_n) / float(data_n)

if __name__ == "__main__":
    # FFNN用
    """
    window = ['window3', 'window5', 'window7', 'multi_window35', 'multi_window37', 'multi_window57', 'multi_window357']

    for w in window:
        base_path = Constant.BASE_PATH + "tsukuba_corpus/LSTM_W2V/" + w + "/zero/summarized_out/softmax/cross_validation"
        for k in range(1, 6):  # 交差検定の番号
            accuracies = []  # 交差検定毎に結果をまとめる
            print("{}分割目".format(k))
            for epoch in range(1, 101):  # エポック数
                print("第{}epoch".format(epoch))
                input_file = base_path + str(k) + "/epoch" + str(epoch) + ".txt"

                accuracy = calculate_accuracy(input_file=input_file)
                accuracies.append(str(accuracy))

            output_file = base_path + str(k) + "/accuracy_file.txt"
            with open(output_file, "w") as f:
                f.write("\n".join(accuracies))
    """

    # LSTM用
    """
    base_path = Constant.BASE_PATH + "tsukuba_corpus/LSTM_W2V/summarized_out/cross_validation"
    for k in range(1, 6):
        accuracies = []  # 交差検定毎に結果をまとめる
        print("{}分割目".format(k))
        for epoch in range(1, 51):  # エポック数
            print("第{}epoch".format(epoch))
            input_file = base_path + str(k) + "/epoch" + str(epoch) + ".txt"

            accuracy = calculate_accuracy(input_file=input_file)
            accuracies.append(str(accuracy))

        output_file = base_path + str(k) + "/accuracy_file.txt"
        with open(output_file, "w") as f:
            f.write("\n".join(accuracies))
    """

    """
    # SVM用
    base_path = Constant.BASE_PATH + "tsukuba_corpus/SVM_BoW/out/"
    accuracies = []
    for k in range(1, 6):  # 交差検定の番号
        input_file = base_path + "cross_validation" + str(k) + ".txt"
        accuracy = calculate_accuracy(input_file=input_file)
        accuracies.append(str(accuracy))


        output_file = base_path + "accuracy_file.txt"
        with open(output_file, "w") as f:
            f.write("\n".join(accuracies))
    """

    # SVMW2V用
    window = ['window3', 'window5', 'window7', 'multi_window35', 'multi_window37', 'multi_window57', 'multi_window357']
    for w in window:
        accuracies = []
        base_path = base_path = Constant.BASE_PATH + "tsukuba_corpus/SVM_W2V/" + w + "/summarized_out/majority/"
        for k in range(1, 6):
            input_file = base_path + "cross_validation" + str(k) + ".txt"
            accuracy = calculate_accuracy(input_file=input_file)
            accuracies.append(str(accuracy))

            output_file = base_path + "accuracy_file.txt"
            with open(output_file, "w") as f:
                f.write("\n".join(accuracies))

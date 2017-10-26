# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def read_accuracy_file(accuracy_file):
    """
    精度ファイルに関して、numpyリストにして返す
    :param accuracy_file: 精度ファイル
    :return: 
    """
    accuracy = []
    with open(accuracy_file, "r") as f:
        for line in f:
            line = line.strip()
            accuracy.append(float(line))
    return np.array(accuracy)


def main():
    # グラフの表示の仕方
    plot_type = 0

    if plot_type == 0:
        # 一つだけ表示させる
        train_accuracy = read_accuracy_file("lstm3_train_accuracy.txt")
        test_accuracy = read_accuracy_file("lstm3_test_accuracy.txt")
        loss = read_accuracy_file("log/loss_log_lstm3.txt")
        n = len(train_accuracy) if len(train_accuracy) > len(test_accuracy) else len(test_accuracy)
        horizon_list = np.array([i for i in range(1, n + 1)])

        # 誤差が無い時、無理やり埋め合わせする
        while(200 > len(loss)):
            loss = np.insert(loss, 0, 0.0)

        # 精度グラフの出力
        fig, ax1 = plt.subplots()
        p1 = ax1.plot(horizon_list, train_accuracy)
        p2 = ax1.plot(horizon_list, test_accuracy)
        ax1.legend((p1[0], p2[0]), ("train accuracy", "test accuracy"), loc=2)
        ax1.set_ylabel('Accuracy')
        plt.ylim(ymax=0.95)

        # 誤差グラフの出力
        ax2 = ax1.twinx()
        ax2.bar(horizon_list, loss, align="center", color="#FF0040", linewidth=0)
        ax2.set_ylabel('Loss')
        plt.ylim(ymax=1000)

        plt.title("LSTM test2")
        plt.grid(True)
        plt.show()

    else:
        # 2つのグラフを並べる
        train_accuracy1 = read_accuracy_file("lstm1_train_accuracy.txt")
        test_accuracy1 = read_accuracy_file("lstm1_test_accuracy.txt")
        train_accuracy2 = read_accuracy_file("lstm2_train_accuracy.txt")
        test_accuracy2 = read_accuracy_file("lstm2_test_accuracy.txt")
        n = len(train_accuracy1) if len(train_accuracy1) > len(test_accuracy1) else len(test_accuracy1)
        horizon_list = np.array([i for i in range(1, n + 1)])

        # グラフ領域のシェア
        fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 4), sharex=True, sharey=True)

        # 1つ目のグラフの描画
        p1 = axL.plot(horizon_list, train_accuracy1)
        p2 = axL.plot(horizon_list, test_accuracy1)
        axL.legend((p1[0], p2[0]), ("train accuracy", "test accuracy"), loc=2)
        axL.set_title('LSTM1')
        axL.set_xlabel('epoch')
        axL.set_ylabel('accuracy')
        axL.grid(True)

        # 2つ目のグラフの描画
        p1 = axR.plot(horizon_list, train_accuracy2)
        p2 = axR.plot(horizon_list, test_accuracy2)
        axR.legend((p1[0], p2[0]), ("train accuracy", "test accuracy"), loc=2)
        axR.set_title('LSTM2')
        axR.set_xlabel('epoch')
        axR.set_ylabel('accuracy')
        axR.grid(True)

        plt.show()


if __name__ == "__main__":
    main()
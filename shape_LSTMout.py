# -*- coding: utf-8 -*-

import Constant


def shape_LSTMout(input_file, output_file):
    # 出力ファイルを初期化
    with open(output_file, "w"):
        pass

    with open(input_file, "r") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 3):
            correct_items = lines[i].strip().split("\t")    # [正解極性, 文]
            predict_items = lines[i+1].strip().split("\t")  # [negativeの確率, positiveの確率]
            predict_polarity = 0 if float(predict_items[0]) > float(predict_items[1]) else 1  # 予測極性

            # 出力ファイルに吐き出す
            with open(output_file, "a") as o:
                o.write(str(correct_items[0]) + "\t" + str(predict_polarity) + "\t" + correct_items[1] + "\n")


if __name__ == "__main__":
    base_path = Constant.BASE_PATH + "tsukuba_corpus/LSTM_W2V/"
    for k in range(1, 6):
        for i in range(1, 51):
            input_file = base_path + "out/cross_validation" + str(k) + "/epoch" + str(i) + ".txt"
            output_file = base_path + "shaped_out/cross_validation" + str(k) + "/epoch" + str(i) + ".txt"
            shape_LSTMout(input_file=input_file, output_file=output_file)

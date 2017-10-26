# -*- coding: utf-8 -*-

"""
FFNN_W2Vの結果をまとめるスクリプト
"""

import Constant


def shape_FFNNout(input_files, output_file):
    """
    FFNNで出した出力をまとめる
    :param input_files: まとめる対象のファイルをリスト形式で
    :param output_file: 出力ファイル名
    :return: なし
    """
    # 出力ファイルの初期化
    with open(output_file, "w"):
        pass

    # 文章数の計算
    sentence_n = 0
    with open(input_files[0], "r") as f:
        for line in f:
            if len(line.strip()) == 0:
                sentence_n += 1

    # ファイルポインタをそれぞれ準備
    file_pointer = []
    for input_file in input_files:
        file_pointer.append(open(input_file))

    # 文章毎に結果をまとめる
    for _ in range(sentence_n):
        # 結果をまとめるリスト
        # sentence_result = [[[文1に対するWindow3の結果], [文1に対するWindow5の結果]], [[文2に対するWindow3の結果], ...], ...]
        sentence_result = []
        for pointer in file_pointer:
            window_result = []
            for line in pointer:
                line = line.strip()
                items = line.split("\t")

                # 空行に差し掛かった
                if len(line) == 0:
                    break

                # 文章の最初の行
                if len(items) == 2:
                    correct_items = items
                    continue

                window_result.append([float(items[0]), float(items[1]), items[2]])
            sentence_result.append(window_result)

        # それぞれの結果から一つに束ねる
        predict_label = consult_majority(sentence_result=sentence_result)

        # 結果ファイルに出力
        with open(output_file, "a") as o:
            o.write(str(correct_items[0]) + "\t" + str(predict_label) + "\t" + correct_items[1] + "\n")

    # ファイルポインタを全てクローズ
    for pointer in file_pointer:
        pointer.close()


def consult_softmax(sentence_result):
    """
    softmaxによってまとめる
    :param sentence_result: 文に対する出力の集合
    :return: 予測極性
    """
    max_abs = 0.0
    predict_label = 0

    # フレーズ毎に参照していく
    for window_result in sentence_result:
        for phrase_result in window_result:
            # 見てるフレーズが、ネガティブの確率の方が高く、その確率値が高い時
            if (phrase_result[0] > phrase_result[1]) and (max_abs < phrase_result[0]):
                max_abs = phrase_result[0]
                predict_label = 0
            # 参照しているフレーズがポジティブの確率の方が高く、その確率値が高い時
            if (phrase_result[0] < phrase_result[1]) and (max_abs < phrase_result[1]):
                max_abs = phrase_result[1]
                predict_label = 1

    return predict_label


def consult_majority(sentence_result):
    """
    多数決によってまとめる
    :param sentence_result: 文に対する出力の集合
    :return: 予測極性
    """
    positive_n = 0  # ポジティブ判定の数
    negative_n = 0  # ネガティブ判定の数

    one_n = 0
    two_n = 0
    three_n = 0
    four_n = 0
    five_n = 0

    # フレーズ毎に参照していく
    for window_result in sentence_result:
        for phrase_result in window_result:
            max_prob = max(phrase_result[0], phrase_result[1], phrase_result[2], phrase_result[3], phrase_result[4])
            if phrase_result[0] == max_prob:
                one_n += 1
            if phrase_result[0] == max_prob:
                two_n += 1
            if phrase_result[0] == max_prob:
                three_n += 1
            if phrase_result[0] == max_prob:
                four_n += 1
            if phrase_result[0] == max_prob:
                five_n += 1
    #         if phrase_result[0] > phrase_result[1]:
    #             negative_n += 1
    #         else:
    #             positive_n += 1
    #
    # if positive_n >= negative_n:
    #     return 1  # ポジティブ判定
    # else:
    #     return 0  # ネガティブ判定

    max_n = max(one_n, two_n, three_n, four_n, five_n)
    if max_n == one_n:
        return 0
    if max_n == two_n:
        return 1
    if max_n == three_n:
        return 2
    if max_n == four_n:
        return 3
    if max_n == five_n:
        return 4

if __name__ == "__main__":
    for k in range(1, 6):  # 交差検定の番号
        print("{}分割目".format(k))
        for epoch in range(1, 101):  # エポック数
            print("第{}epoch".format(epoch))
            input_file1 = Constant.BASE_PATH + "tsukuba_corpus/FFNN_W2V/Window3/exception/out/cross_validation" + str(k) + "/epoch" + str(epoch) + ".txt"
            input_file2 = Constant.BASE_PATH + "tsukuba_corpus/FFNN_W2V/Window5/exception/out/cross_validation" + str(k) + "/epoch" + str(epoch) + ".txt"
            input_file3 = Constant.BASE_PATH + "tsukuba_corpus/FFNN_W2V/Window7/exception/out/cross_validation" + str(k) + "/epoch" + str(epoch) + ".txt"

            output_file_path1 = Constant.BASE_PATH + "tsukuba_corpus/FFNN_W2V/"
            output_file_path2 = "/exception/shaped_out/majority/cross_validation" + str(k) + "/epoch" + str(epoch) + ".txt"


            shape_FFNNout(input_files=[input_file1], output_file=output_file_path1+"Window3"+output_file_path2)
            shape_FFNNout(input_files=[input_file2], output_file=output_file_path1 + "Window5" + output_file_path2)
            shape_FFNNout(input_files=[input_file3], output_file=output_file_path1 + "Window7" + output_file_path2)

            shape_FFNNout(input_files=[input_file1, input_file2], output_file=output_file_path1 + "MultiWindow35" + output_file_path2)
            shape_FFNNout(input_files=[input_file1, input_file3], output_file=output_file_path1 + "MultiWindow37" + output_file_path2)
            shape_FFNNout(input_files=[input_file2, input_file3], output_file=output_file_path1 + "MultiWindow57" + output_file_path2)
            shape_FFNNout(input_files=[input_file1, input_file2, input_file3], output_file=output_file_path1 + "MultiWindow357" + output_file_path2)

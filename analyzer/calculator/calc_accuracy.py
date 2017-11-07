# -*- coding: utf-8 -*-

import re
import math


def calc_accuracy_1time(input_file, n_label):
    """
    1回分の正答率を計算して、結果を返す
    :param input_file: 結果ファイル
    :param n_label: 何値分類であるか
    :return: [正答率, ラベル1に対する正答率, ...ラベルn_labelに対する正答率]
    """
    label_correct_n = [0.0 for _ in range(n_label)]  # ラベル毎の正解数(最後に正答率を計算するのでfloat)
    label_wrong_n = [0.0 for _ in range(n_label)]  # ラベル毎の不正解数(最後に正答率を計算するのでfloat)

    with open(input_file) as f:
        for line in f:
            items = line.split('\t')  # [正解ラベル, 予測ラベル, 文章]
            if items[0] == items[1]:
                label_correct_n[int(items[0])] += 1.0
            else:
                label_wrong_n[int(items[0])] += 1.0

    accuracy = list()
    # 全体の正答率を計算
    accuracy.append(sum(label_correct_n) / (sum(label_correct_n) + sum(label_wrong_n)))

    # 各ラベルに対する正答率を計算
    for i in range(n_label):
        # あるラベルに対して、データがなかった場合は、0とする
        if (label_correct_n[i] + label_wrong_n[i]) == 0.0:
            accuracy.append(0.0)
            continue
        accuracy.append(label_correct_n[i] / (label_correct_n[i] + label_wrong_n[i]))

    return accuracy


def calc_accuracy_epochs(input_files, output_file, n_label):
    """
    正答率を計算し、ファイルに出力する
    :param input_files: エポック毎の結果がまとめてあるファイルリスト
    :param output_file: 出力ファイル
    :param n_label: 何値分類であるか
    :return: なし
    """
    # 出力ファイルを初期化する
    with open(output_file, 'w'):
        pass

    # ファイル名のみを取り出す正規表現
    pattern = r"epoch.*"

    for input_file in input_files:
        accuracy = calc_accuracy_1time(input_file, n_label)  # 正答率はfloatで返ってくる
        accuracy = [str(data) for data in accuracy]  # str化

        m = re.search(pattern, input_file)
        if m:
            with open(output_file, 'a') as o:
                o.write(m.group() + ':' + '\t'.join(accuracy) + '\n')


def calc_mean_square_error_1time(input_file):
    """
    2乗誤差を計算して、結果を返す
    :param input_file: 結果ファイル
    :return: 正答率
    """
    sum_mean_square_error = 0.0  # 合計2乗誤差
    sentence_n = 0.0  # 全体の文章数

    with open(input_file) as f:
        for line in f:
            sentence_n += 1.0
            items = line.split('\t')  # [正解ラベル, 予測ラベル, 文章]
            sum_mean_square_error += math.sqrt((float(items[0]) - float(items[1])) ** 2)

    return sum_mean_square_error / sentence_n


def calc_mean_square_error_epochs(input_files, output_file):
    """
    2乗誤差を計算し、ファイルに出力する
    :param input_files: エポック毎の結果がまとめてあるファイルリスト
    :param output_file: 出力ファイル
    :return: なし
    """
    # 出力ファイルを初期化する
    with open(output_file, 'w'):
        pass

    # ファイル名のみを取り出す正規表現
    pattern = r"epoch.*"

    for input_file in input_files:
        accuracy = calc_mean_square_error_1time(input_file)
        m = re.search(pattern, input_file)
        if m:
            with open(output_file, 'a') as o:
                o.write(m.group() + ':\t' + str(accuracy) + '\n')


def calc_accuracy_cross_validation(input_files, output_file):
    """
    正答率を計算し、ファイルに出力する
    :param input_files: 交差検定の結果がまとめてあるファイルリスト
    :param output_file: 出力ファイル
    :return: なし
    """
    # 出力ファイルを初期化する
    with open(output_file, 'w'):
        pass

    # ファイル名のみを取り出す正規表現
    pattern = r"cross_validation1.*"

    for input_file in input_files:
        accuracy = calc_accuracy_1time(input_file=input_file)
        m = re.search(pattern, input_file)
        if m:
            with open(output_file, 'a') as o:
                o.write(m.group() + ':\t' + str(accuracy) + '\n')

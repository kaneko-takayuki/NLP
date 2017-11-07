# -*- coding: utf-8 -*-

import re
import math


def calc_accuracy_1time(input_file, n_label):
    """
    1回分の正答率を計算して、結果を返す
    :param input_file: 結果ファイル
    :param n_label: 何値分類であるか
    :return: [全体の正答率, ラベル1に対する正答率, ...ラベルn_labelに対する正答率]
    """
    label_correct_n = [0.0 for _ in range(n_label)]  # ラベル毎の正解数(最後に正答率を計算するのでfloat)
    label_wrong_n = [0.0 for _ in range(n_label)]  # ラベル毎の不正解数(最後に正答率を計算するのでfloat)

    with open(input_file) as f:
        for line in f:
            items = line.split('\t')  # [正解ラベル, 予測ラベル, 文章]
            if items[0] == items[1]:
                label_correct_n[int(items[0])] += 1
            else:
                label_wrong_n[int(items[0])] += 1

    accuracy = list()
    # 全体の正答率を計算
    accuracy.append(sum(label_correct_n) / (sum(label_correct_n) + sum(label_wrong_n)))

    # 各ラベルに対する正答率を計算
    for i in range(n_label):
        # あるラベルに対して、データがなかった場合は、0とする
        if (label_correct_n[i] + label_wrong_n[i]) == 0.0:
            accuracy.append('Label' + str(i) + '_Nothing')
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


def calc_mean_square_error_1time(input_file, n_label):
    """
    2乗誤差を計算して、結果を返す
    :param input_file: 結果ファイル
    :param n_label: 何値分類か
    :return: [全体の平均2乗誤差, ラベル1に対する平均2乗誤差, ...ラベルn_labelに対する平均2乗誤差]
    """
    sum_mean_square_error = [0.0 for _ in range(n_label)]  # ラベル毎の合計2乗誤差(最後に正答率を計算するのでfloat)
    sentence_n = [0.0 for _ in range(n_label)]  # ラベル毎の文章数(最後に正答率を計算するのでfloat)

    with open(input_file) as f:
        for line in f:
            items = line.split('\t')  # [正解ラベル, 予測ラベル, 文章]
            sentence_n[int(items[0])] += 1
            sum_mean_square_error[int(items[0])] += math.sqrt((float(items[0]) - float(items[1])) ** 2)

    square_error = list()
    # 全体の平均2乗誤差を計算
    square_error.append(sum(sum_mean_square_error) / sum(sentence_n))

    # 各ラベル毎の平均2乗誤差を計算
    for i in range(n_label):
        if sentence_n[i] == 0.0:
            square_error.append('Label'+str(i)+'_Nothing')
            continue
        square_error.append(sum_mean_square_error[i] / sentence_n[i])

    return square_error


def calc_mean_square_error_epochs(input_files, output_file, n_label):
    """
    2乗誤差を計算し、ファイルに出力する
    :param input_files: エポック毎の結果がまとめてあるファイルリスト
    :param output_file: 出力ファイル
    :param n_label: 何値分類か
    :return: なし
    """
    # 出力ファイルを初期化する
    with open(output_file, 'w'):
        pass

    # ファイル名のみを取り出す正規表現
    pattern = r"epoch.*"

    for input_file in input_files:
        accuracy = calc_mean_square_error_1time(input_file, n_label)  # floatで返ってくる
        accuracy = [str(data) for data in accuracy]  # str化
        m = re.search(pattern, input_file)
        if m:
            with open(output_file, 'a') as o:
                o.write(m.group() + ':' + '\t'.join(accuracy) + '\n')


def calc_accuracy_cross_validation(input_files, output_file, n_label):
    """
    正答率を計算し、ファイルに出力する
    :param input_files: 交差検定の結果がまとめてあるファイルリスト
    :param output_file: 出力ファイル
    :param n_label: 何値分類か
    :return: なし
    """
    # 出力ファイルを初期化する
    with open(output_file, 'w'):
        pass

    # ファイル名のみを取り出す正規表現
    pattern = r"cross_validation1.*"

    for input_file in input_files:
        accuracy = calc_accuracy_1time(input_file, n_label)
        accuracy = [str(data) for data in accuracy]
        m = re.search(pattern, input_file)
        if m:
            with open(output_file, 'a') as o:
                o.write(m.group() + ':' + '\t'.join(accuracy) + '\n')

# -*- coding: utf-8 -*-

import re


def calc_accuracy_1time(input_file):
    """
    1回分の正答率を計算して、結果を返す
    :param input_file: 結果ファイル
    :return: 正答率
    """
    correct_n = 0.0  # 正解数
    wrong_n = 0.0  # 不正解数

    with open(input_file) as f:
        for line in f:
            items = line.split('\t')  # [正解ラベル, 予測ラベル, 文章]
            if items[0] == items[1]:
                correct_n += 1
            else:
                wrong_n += 1

    return correct_n / (correct_n + wrong_n)


def calc_accuracy_epochs(input_files, output_file):
    """
    正答率を計算し、ファイルに出力する
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
        accuracy = calc_accuracy_1time(input_file)
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
    pattern = r"cross_validation.*"

    for input_file in input_files:
        accuracy = calc_accuracy_1time(input_file=input_file)
        m = re.search(pattern, input_file)
        if m:
            with open(output_file, 'a') as o:
                o.write(m.group() + ':\t' + str(accuracy) + '\n')

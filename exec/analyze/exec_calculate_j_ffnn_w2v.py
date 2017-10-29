# -*- coding: utf-8 -*-

import os
import sys
import glob
import functools
import add_path
from analyzer.calculator import calc_accuracy


def calculate_accuracy(experiment_dir):
    """
    まとめた結果ファイルに対して、精度を計算してファイルに出力
    :param experiment_dir: 実験結果ファイルが格納されているディレクトリ
    :return: なし
    """
    for i in range(1, 6):
        k_experiment_dir = experiment_dir + "cross_validation" + str(i) + "/"
        input_files = glob.glob(k_experiment_dir + "epoch*")
        input_files = sorted(input_files, key=functools.cmp_to_key(str_compare))
        # 精度を計算
        calc_accuracy.calc_accuracy_epochs(input_files, k_experiment_dir + "accuracy_file2.txt")


def str_compare(str1, str2):
    """
    文字列の比較
    :param str1:
    :param str2:
    :return:
    """
    if len(str1) < len(str2):
        return len(str1) - len(str2)
    else:
        if str1 < str2:
            return -1
        else:
            return 1


def main():
    # プロジェクトディレクトリを取得
    base_path = os.path.dirname(os.path.abspath(__file__)) + "/../../"

    # つくばコーパスについて
    """
    使用モデル: {FFNN, SVM}
    ウィンドウサイズ: {3, 5, 7, 3+5, 3+7, 5+7, 3+5+7}
    補完関数: {zero, random, exception}
    合議関数: {多数決, softmax}
    """
    ws_list = ["window3", "window5", "window7",
               "multi_window35", "multi_window37", "multi_window57", "multi_window357"]
    completion_list = ["zero", "random", "exception"]
    summarize_list = ["majority", "softmax"]

    for ws in ws_list:
        for completion in completion_list:
            for summarize in summarize_list:
                experiment_dir = base_path + "tsukuba_corpus/ffnn_w2v/" + ws + \
                                 "/" + completion + "/" + "summarized_out/" + summarize + "/"
                print("-------------------------")
                print("モデル: ffnn")
                print("ウィンドウサイズ: " + ws)
                print("補完関数: " + completion)
                print("合議関数: " + summarize)
                print("-------------------------")
                sys.stdout.write("精度を計算中...")
                sys.stdout.flush()
                calculate_accuracy(experiment_dir=experiment_dir)
                print("完了\n")


if __name__ == '__main__':
    main()

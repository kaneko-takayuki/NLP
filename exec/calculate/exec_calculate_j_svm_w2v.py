# -*- coding: utf-8 -*-

import os
import sys
import add_path
from analyzer.calculator import calc_accuracy


def calculate_accuracy(experiment_dir):
    """
    まとめた結果ファイルに対して、精度を計算してファイルに出力
    :param experiment_dir: 実験結果ファイルが格納されているディレクトリ
    :param output_file: 出力ファイル
    :return: なし
    """
    input_files = []
    for i in range(1, 6):
        input_files.append(experiment_dir + "cross_validation" + str(i) + ".txt")

    # 精度を計算
    calc_accuracy.calc_accuracy_cross_validation(input_files, experiment_dir + "accuracy_file2.txt")


def main():
    # プロジェクトディレクトリを取得
    base_path = os.path.dirname(os.path.abspath(__file__)) + "/../../"

    # つくばコーパスについて
    """
    使用モデル: {SVM}
    ウィンドウサイズ: {3, 5, 7, 3+5, 3+7, 5+7, 3+5+7}
    補完関数: {random}
    合議関数: {多数決, softmax}
    """
    ws_list = ["window3", "window5", "window7",
               "multi_window35", "multi_window37", "multi_window57", "multi_window357"]
    summarize_list = ["majority", "softmax"]

    for ws in ws_list:
        for summarize in summarize_list:
            experiment_dir = base_path + "tsukuba_corpus/svm_w2v/" + ws + \
                             "/" + "summarized_out/" + summarize + "/"
            print("-----------------------------------")
            print("モデル: svm")
            print("ウィンドウサイズ: " + ws)
            print("補完関数: random")
            print("合議関数: " + summarize)
            print("-----------------------------------")
            sys.stdout.write("精度を計算中...")
            sys.stdout.flush()
            calculate_accuracy(experiment_dir=experiment_dir)
            print("完了\n")


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-

import argparse
import add_path
from analyzer.calculator import calc_accuracy


def main(experiment_dir):
    """
    日本語SVMの出力について、精度を計算する
    :param experiment_dir: 実験ディレクトリ
    :return: なし
    """
    # パラメータの表示
    print("合議対象:")
    print("\t- 日本語文\n\t- 2段階評価\n\t- SVM")
    print("対象ディレクトリ:")
    print('\t- ' + experiment_dir)
    print("-----------------------------------")

    # 計算するするファイルリストを用意
    input_files = [experiment_dir + "cross_validation1" + str(i) + ".txt" for i in range(1, 6)]

    # 精度を計算
    calc_accuracy.calc_accuracy_cross_validation(input_files, experiment_dir + "accuracy_file.txt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SVMの合議による2値分類の精度を計算する')
    parser.add_argument("--experiment_dir", "-d", type=str, help='実験結果が格納されているディレクトリ')
    args = parser.parse_args()

    main(experiment_dir=args.experiment_dir)

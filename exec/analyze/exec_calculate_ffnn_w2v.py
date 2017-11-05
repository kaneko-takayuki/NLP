# -*- coding: utf-8 -*-

import argparse
from analyzer.calculator import calc_accuracy


def main(experiment_dir, n_epoch):
    """
    日本語FFNNの出力について、精度を計算する
    :param experiment_dir: 実験ディレクトリ
    :param n_epoch: エポック数
    :return: なし
    """
    # パラメータの表示
    print("合議対象:")
    print("\t- 日本語文\n\t- 2段階評価\n\t- FFNN")
    print("対象ディレクトリ:")
    print('\t- ' + experiment_dir)
    print("-----------------------------------")

    # 計算するファイルリストを用意
    input_files = [experiment_dir + "epoch" + str(i) + ".tsv" for i in range(1, n_epoch+1)]

    # 精度を計算
    calc_accuracy.calc_accuracy_epochs(input_files, experiment_dir + "accuracy_file.txt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FFNNの合議による2値分類の精度を計算する')
    parser.add_argument("--experiment_dir", "-d", type=str, help='合議結果が格納されているディレクトリ')
    parser.add_argument("--n_epoch", "-e", type=int, default=50, help='どのエポックまでの結果をまとめるか')
    args = parser.parse_args()

    main(experiment_dir=args.experiment_dir, n_epoch=args.n_epoch)

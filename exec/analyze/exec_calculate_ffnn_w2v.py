# -*- coding: utf-8 -*-

import argparse
import add_path
from analyzer.calculator import calc_accuracy


def main(experiment_dir, n_epoch, n_label):
    """
    合議出力について、精度を計算する
    :param experiment_dir: 実験ディレクトリ
    :param n_epoch: エポック数
    :param n_label: 何値分類であるか
    :return: なし
    """
    # パラメータの表示
    print("対象ディレクトリ:")
    print('\t- ' + experiment_dir)
    print("-----------------------------------")

    # 計算するファイルリストを用意
    input_files = [experiment_dir + "epoch" + str(i) + ".tsv" for i in range(1, n_epoch+1)]

    # 精度を計算
    calc_accuracy.calc_accuracy_epochs(input_files, experiment_dir + "accuracy_file.txt", n_label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='n値分類の結果の精度を計算する')
    parser.add_argument("--experiment_dir", "-d", type=str, help='合議結果が格納されているディレクトリ')
    parser.add_argument("--n_epoch", "-e", type=int, default=50, help='どのエポックまでの結果をまとめるか')
    parser.add_argument("--n_label", "-l", type=int, default=5, help='何値分類であるか')
    args = parser.parse_args()

    main(experiment_dir=args.experiment_dir, n_epoch=args.n_epoch, n_label=args.n_label)

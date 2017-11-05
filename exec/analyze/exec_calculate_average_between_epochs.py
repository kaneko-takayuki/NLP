# -*- coding: utf-8 -*-

import argparse


def main(experiment_dir, start_epoch, end_epoch):
    """
    精度ファイルについて、指定したエポック区間の平均精度を出力する
    :param experiment_dir: 実験ディレクトリ
    :param start_epoch: 開始エポック数
    :param end_epoch: 終了エポック
    :return: なし
    """
    # パラメータの表示
    print("対象ディレクトリ:")
    print('\t- ' + experiment_dir)
    print("-----------------------------------")

    sum_accuracy = 0.0
    with open(experiment_dir + "accuracy_file.txt") as f:
        for i, line in enumerate(f):
            if start_epoch <= i < end_epoch:
                sum_accuracy += float(line.split('\t')[1])

    print(str(sum_accuracy / (end_epoch+1 - start_epoch)))

    print("-----------------------------------\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FFNNの合議による2値分類の精度を計算する')
    parser.add_argument("--experiment_dir", "-d", type=str, help='合議結果が格納されているディレクトリ')
    parser.add_argument("--start_epoch", "-se", type=int, default=50, help='どのエポックから')
    parser.add_argument("--end_epoch", "-ee", type=int, default=50, help='どのエポックまでの結果をまとめるか')
    args = parser.parse_args()

    main(experiment_dir=args.experiment_dir, start_epoch=args.start_epoch, end_epoch=args.end_epoch)

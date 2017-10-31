# -*- coding: utf-8 -*-

import argparse
import add_path
from analyzer import summarizer
from analyzer.summarizer import e_sigmoid5_w2v
from analyzer.summarizer import consult_function as cf


def main(input_dirs, output_dir, consult, n_epoch):
    """
    英語sigmoid5のウィンドウサイズ毎の出力を合議によってまとめる
    :param input_dirs: (ウィンドウサイズの)出力がまとめてあるディレクトリ
    :param output_dir: 出力ディレクトリ
    :param consult: 合議手法
    :param n_epoch: どのエポックまで合議を行うか
    :return: なし
    """
    # パラメータの表示
    print("合議対象:")
    print("\t- 英文\n\t- 5段階評価\t- sigmoid5")
    print("入力ディレクトリ:")
    for dir_name in input_dirs:
        print('\t- ' + dir_name)
    print("出力ディレクトリ:")
    print('\t- ' + output_dir)
    print("合議手法:")
    print('\t- ' + consult)
    print("-----------------------------------")

    # 合議関数を設定する
    if consult == "majority":
        summarizer.consult_func = cf.sigmoid5_consult_majority
    else:
        summarizer.consult_func = cf.sigmoid5_consult_softmax

    # エポック毎に合議を行う
    for e in range(1, n_epoch+1):
        input_files = [(dir_name + "epoch" + str(e) + ".tsv") for dir_name in input_dirs]
        e_sigmoid5_w2v.summarize(input_files, output_dir + "epoch" + str(e) + "tsv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='summarize ffnn-out to english five classification with majority')
    parser.add_argument("--input_dirs", "-i", nargs='*')
    parser.add_argument("--output_dir", "-o")
    parser.add_argument("--consult", "-c")
    parser.add_argument("--n_epoch", "-e")
    args = parser.parse_args()

    main(input_dirs=args.input_files, output_dir=args.output_file, consult=args.consult, n_epoch=args.n_epoch)

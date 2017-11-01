# -*- coding: utf-8 -*-

import argparse
import add_path
from analyzer import summarizer
from analyzer.summarizer import e_ffnn_w2v
from analyzer.summarizer import consult_function as cf


def main(input_dirs, output_dir, consult, n_epoch):
    """
    英語FFNNのウィンドウサイズ毎の出力を合議によってまとめる
    :param input_dirs: (ウィンドウサイズの)出力がまとめてあるディレクトリ
    :param output_dir: 出力ディレクトリ
    :param consult: 合議手法
    :param n_epoch: どのエポックまで合議を行うか
    :return: なし
    """
    # パラメータの表示
    print("合議対象:")
    print("\t- 英文\n\t- 5段階評価\n\t- FFNN")
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
        summarizer.consult_func = cf.ffnn_consult_majority5
    else:
        summarizer.consult_func = cf.ffnn_consult_softmax5

    # エポック毎に合議を行う
    for e in range(1, n_epoch+1):
        input_files = [(dir_name + "epoch" + str(e) + ".tsv") for dir_name in input_dirs]
        e_ffnn_w2v.summarize(input_files, output_dir + "epoch" + str(e) + "tsv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Amazonコーパスについて、FFNNによって出力した結果を合議する')
    parser.add_argument("--input_dirs", "-i", nargs='*', help='エポック毎の結果が格納されているディレクトリ(複数指定可能)')
    parser.add_argument("--output_dir", "-o", type=str, help='結果を出力するディレクトリ')
    parser.add_argument("--consult", "-c", type=str, default='majority', help='合議方法')
    parser.add_argument("--n_epoch", "-e", type=int, default=50, help='どのエポックまでの結果をまとめるか')
    args = parser.parse_args()

    main(input_dirs=args.input_dirs, output_dir=args.output_dir, consult=args.consult, n_epoch=args.n_epoch)

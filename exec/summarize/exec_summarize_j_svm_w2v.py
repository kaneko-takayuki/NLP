# -*- coding: utf-8 -*-

import argparse
import add_path
from analyzer import summarizer
from analyzer.summarizer import j_svm_w2v
from analyzer.summarizer import consult_function as cf


def main(input_dirs, output_dir, consult, cross_k):
    """
    日本語SVMのウィンドウサイズ毎の出力を合議によってまとめる
    :param input_dirs: (ウィンドウサイズの)出力がまとめてあるディレクトリ
    :param output_dir: 出力ディレクトリ
    :param consult: 合議手法
    :param cross_k: k分割目について合議を行う
    :return: なし
    """
    # パラメータの表示
    print("合議対象:")
    print("\t- 日本語文\n\t- 2段階評価\n\t- SVM")
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
        summarizer.consult_func = cf.svm_consult_majority
    else:
        summarizer.consult_func = cf.svm_consult_softmax

    # 合議を行う
    input_files = [(dir_name + "cross_validation" + str(cross_k) + ".tsv") for dir_name in input_dirs]
    j_svm_w2v.summarize(input_files, output_dir + "cross_validation" + str(cross_k) + "tsv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='summarize svm-out to japanese binary classification')
    parser.add_argument("--input_dirs", "-i", nargs='*')
    parser.add_argument("--output_dir", "-o")
    parser.add_argument("--consult", "-c")
    parser.add_argument("--cross_k", "-k")
    args = parser.parse_args()

    main(input_dirs=args.input_dirs, output_dir=args.output_dir, consult=args.consult, cross_k=args.cross_k)
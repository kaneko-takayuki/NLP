# -*- coding: utf-8 -*-

import argparse
import add_path
from analyzer import summarizer
from analyzer.summarizer import j_ffnn_w2v
from analyzer.summarizer import consult_function as cf


def main(input_files, output_file):
    """
    日本語用FFNNで出力した結果を多数決によってまとめる
    :param input_files: 入力ファイルリスト
    :param output_file: 出力ファイル
    :return: なし
    """
    # 合議関数を設定
    summarizer.consult_func = cf.ffnn_consult_majority
    # 実行
    j_ffnn_w2v.summarize_ffnn_w2v(input_files, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='summarize ffnn-out to japanese binary classification with majority')
    parser.add_argument("--input_files", "-i", nargs='*')
    parser.add_argument("--output_file", "-o")
    args = parser.parse_args()

    main(input_files=args.input_files, output_file=args.output_file)

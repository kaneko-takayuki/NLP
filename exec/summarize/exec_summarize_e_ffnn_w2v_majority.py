# -*- coding: utf-8 -*-

import os
import argparse
import add_path
from analyzer import summarizer
from analyzer.summarizer import e_ffnn_w2v
from analyzer.summarizer import consult_function as cf


def summarize(input_files, output_file):
    """
    英語用FFNNで出力した結果を多数決によってまとめる
    :param input_files: 入力ファイルリスト
    :param output_file: 出力ファイル
    :return: なし
    """
    # 合議関数を設定
    summarizer.consult_func = cf.ffnn_consult_majority5
    # 実行
    e_ffnn_w2v.summarize(input_files, output_file)


def main(input_files, output_file):
    base_path = os.path.dirname(os.path.abspath(__file__)) + "/../../"
    _window = ["Window3", "Window5", "Window7", "Window9", "Window11", "Window13"]

    for w in _window:
        for i in range(1, 6):
            print("window_size: " + w)
            print("cross_validation: " + str(i))
            print("------------------------------")
            for j in range(1, 101):
                experiment_dir = base_path + "amazon_corpus/FFNN_W2V/" + w + "/random/"
                input_file = experiment_dir + "out/cross_validation" + str(i) + "/epoch" + str(j) + ".txt"
                output_file = experiment_dir + "summarized_out/majority/cross_validation" + str(i) + "/epoch" + str(j) + ".tsv"
                summarize([input_file], output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='summarize ffnn-out to english binary classification with majority')
    parser.add_argument("--input_files", "-i", nargs='*')
    parser.add_argument("--output_file", "-o")
    args = parser.parse_args()

    main(input_files=args.input_files, output_file=args.output_file)

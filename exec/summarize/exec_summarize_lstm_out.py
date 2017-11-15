# -*- coding: utf-8 -*-

import sys
import argparse
import add_path
from amazon_corpus import functions
from ml.svm.lstm_svm import LSTMSVM as lstm_svm


def main(dir_name, n_epoch):
    """
    :param dir_name: 実験ディレクトリ
    :param n_epoch: どのエポックまでまとめるか
    :return: なし
    """
    # エポック毎にまとめる
    for e in range(1, n_epoch + 1):
        sys.stdout.write("epoch" + str(e) + "...")
        sys.stdout.flush()
        input_dir = dir_name + "out/cross_validation1/"
        output_file = dir_name + "summarized_out/cross_validation1/epoch" + str(e) + ".tsv"
        model = lstm_svm()

        # 学習データの読み込み
        train_sentences = []
        train_vectors = []
        train_labels = []
        for i in range(1, 5):
            data = functions.read_amazon_lstm_w2v(input_dir + "epoch" + str(e) + "_train" + str(i) + ".tsv")
            train_sentences.extend(data[0])
            train_vectors.extend(data[1])
            train_labels.extend(data[2])
        model.set_train_data(train_sentences, train_vectors, train_labels)

        # テストデータの読み込み
        data = functions.read_amazon_lstm_w2v(input_dir + "epoch" + str(e) + "_test.tsv")
        model.set_test_data(data[0], data[1], data[2])

        model.train()
        model.test(output_file)
        
        print("完了")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Amazonコーパスについて、sigmoid5によって出力した結果を合議する')
    parser.add_argument("--dir", "-d", help='実験ディレクトリ')
    parser.add_argument("--n_epoch", "-e", type=int, default=50, help='どのエポックまでの結果をまとめるか')
    args = parser.parse_args()

    main(args.dir, args.n_epoch)

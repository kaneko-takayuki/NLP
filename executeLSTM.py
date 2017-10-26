# -*- coding: utf-8 -*-

import sys
import argparse

import LSTMRunner


def prepare_data(file_name):
    """
    file_nameを読み込み、リストで返す
    :param file_name: 
    :return: [[文1, ラベル1], [文2, ラベル2], ..., [文N, ラベルN]]
    """
    data = []
    with open(file_name, "r") as f:
        for line in f:
            items = line.split("\t")
            label = 1 if items[4] == "p" else 0
            sentence = items[5]
            data.append([sentence, label])

    return data


def read_amazon_data(file_name):
    data = []
    with open(file_name, "r") as f:
        for line in f:
            review = line.split("\t")
            data.append([review[1], review[0]])
    return data


def execute_lstm(n_in=900, n_units=1000, n_out=2, gpu=-1, n_epoch=10, batchsize=30):
    # ネットワークインスタンス作成
    net = LSTMRunner.LSTMRunner()
    net.init_model(n_in=n_in,
                     n_units=n_units,
                     n_out=n_out,
                     gpu=gpu)

    # 5分割交差検定
    for k in range(1, 6):
        net.init_model(n_in=n_in,
                     n_units=n_units,
                     n_out=n_out,
                     gpu=gpu)
        # 学習データ
        train_data = []
        # テストデータ
        test_data = []
        for j in range(1, 6):
            if k == j:
                test_data.extend(prepare_data("tsukuba_corpus/unprocessed_data/dataset" + str(j) + ".tsv"))
            else:
                train_data.extend(prepare_data("tsukuba_corpus/unprocessed_data/dataset" + str(j) + ".tsv"))

        # エポックを回す
        for epoch in range(1, n_epoch+1):
            sys.stdout.write("epoch" + str(epoch) + "...")
            net.set_train_data(train_data)
            net.set_test_data(test_data)
            net.train(batchsize=batchsize)
            net.test("tsukuba_corpus/LSTM_W2V/out/cross_validation" + str(k) + "/epoch" + str(epoch) + ".txt")
            net.save("tsukuba_corpus/LSTM_W2V/model/cross_validation" + str(k) + "/epoch" + str(epoch) + "_model.npz")
            print("完了")

    net.send_mail("LSTM", "LSTMの実験が終わりました！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM_parameter')
    parser.add_argument("--gpu", "-g", type=int, default=-1)
    parser.add_argument("--nin", "-i", type=int, default=300)
    parser.add_argument("--hidden", "-m", type=int, default=1000)
    parser.add_argument("--out", "-o", type=int, default=2)
    parser.add_argument("--epoch_n", "-e", type=int, default=20)
    parser.add_argument("--batchsize", "-b", type=int, default=30)
    args = parser.parse_args()

    execute_lstm(n_in=args.nin,
                 n_units=args.hidden,
                 n_out=args.out,
                 gpu=args.gpu,
                 n_epoch=args.epoch_n,
                 batchsize=args.batchsize)


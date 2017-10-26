# -*- coding: utf-8 -*-

import sys
import argparse

import FFNNConsultation as ffnn
import SentenceConvertor


def prepare_data(file_name, window_size):
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
            if SentenceConvertor.SentenceConvertor(sentence=sentence).get_word_count() < window_size:
                continue
            data.append([sentence, label])

    return data


def prepare_amazon_data(file_name):
    data = []
    with open(file_name, "r") as f:
        for line in f:
            review = line.split("\t")
            label = int(review[0]) - 1
            data.append([review[1], label])
    return data


def execute_ffnn(window_size=3, n_in=900, n_units=1000, n_out=5, gpu=-1, n_epoch=20, batchsize=30):
    # ネットワークインスタンス作成
    net = ffnn.FFNNConsultation()

    # 5分割交差検定
    for k in range(1, 6):
        # 学習データ
        train_data = []
        # テストデータ
        test_data = []
        for j in range(1, 6):
            # ネットワークの初期化
            net.init_model(n_in=n_in, n_units=n_units, n_out=n_out, window_size=window_size, gpu=gpu)
            if k == j:
                test_data.extend(prepare_amazon_data(file_name="amazon_corpus/data/books/dataset" + str(j) + ".tsv"))
            else:
                train_data.extend(prepare_amazon_data(file_name="amazon_corpus/data/books/dataset" + str(j) + ".tsv"))

        # エポックを回す
        for epoch in range(1, n_epoch+1):
            sys.stdout.write("epoch" + str(epoch) + "...")
            net.set_train_data(train_data)
            net.set_test_data(test_data)
            net.train(batchsize=batchsize)
            net.test("amazon_corpus/FFNN_W2V/Window" + str(window_size) + "/random/out/cross_validation" + str(k) + "/epoch" + str(epoch) + ".txt")
            net.save("amazon_corpus/FFNN_W2V/Window" + str(window_size) + "/random/model/cross_validation" + str(k) + "/epoch" + str(epoch) + "_model.npz")
            print("完了")

    net.send_mail("FFNN", "ウィンドウサイズ" + str(window_size) + "の実験が終わりました！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FFNN_parameter')
    parser.add_argument("--windowsize", "-w", type=int, default=3)
    parser.add_argument("--gpu", "-g", type=int, default=-1)
    parser.add_argument("--in", "-i", type=int, default=900)
    parser.add_argument("--hidden", "-m", type=int, default=1000)
    parser.add_argument("--out", "-o", type=int, default=2)
    parser.add_argument("--epoch_n", "-e", type=int, default=20)
    parser.add_argument("--batchsize", "-b", type=int, default=30)
    args = parser.parse_args()
    execute_ffnn(window_size=args.windowsize,
                 n_in=args.windowsize*300,
                 n_units=args.hidden,
                 n_out=args.out,
                 gpu=args.gpu,
                 n_epoch=args.epoch_n,
                 batchsize=args.batchsize)

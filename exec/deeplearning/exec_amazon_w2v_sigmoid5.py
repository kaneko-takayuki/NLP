# -*- coding: utf-8 -*-

import os
import sys
import argparse

from econvertor.word2vec import functions as w2v_func
from ml.deeplearning.e_w2v_sigmoid5 import EW2VSigmoid5
from amazon_corpus.functions import read_amazon_corpus
import constants


def main(start_k, end_k, start_epoch, end_epoch, n_in, n_mid, batchsize, gpu, window_size, completion):
    """
    Amazonコーパスに対して、
    sigmoidを5つ使用したモデルで、フレーズベクトルを素性として、学習・テストを行う
    :param start_k: 5分割交差検定において、どこから行うか
    :param end_k: 5分割交差検定において、どこまで行うか
    :param start_epoch: 開始エポック数
    :param end_epoch: 終了エポック数
    :param n_in: 入力次元数
    :param n_mid: 中間次元数
    :param batchsize: バッチサイズ
    :param gpu: GPUを利用するかどうか
    :param window_size: フレーズを区切るウィンドウサイズ
    :param completion: 補完関数(zero, random)
    :return: なし
    """

    # pycharmから動かすとcudaへのパスが通らないので、ここでパスを追加する
    os.environ["PATH"] = "/usr/local/cuda-7.5/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

    # 実験ディレクトリ
    experiment_dir = "amazon_corpus/sigmoid_w2v/window" + str(window_size) + "/random/"

    # 実験で使用する補完関数を設定
    if completion == "zero":
        w2v_func.set_completion_func(w2v_func.create_zero_vector)
    elif completion == "random":
        w2v_func.set_completion_func(w2v_func.create_random_vector)
    else:
        sys.stderr.write("指定した補完関数が適切ではありません\n")
        exit()

    # 実験で使用するword2vecモデルを読み込む
    w2v_func.load_w2v("/home/kaneko-takayuki/NLP/w2v_model/GoogleNews-vectors-negative300.bin")

    # k_start〜k_endで5分割交差検定
    # k: k回目の検定
    for k in range(start_k, end_k):
        print("-------------------")
        print(str(k) + " / 5 分割目")
        print("-------------------")
        # ネットワークインスタンス作成
        net = EW2VSigmoid5(n_in, n_mid, batchsize, gpu, window_size)

        # 途中のエポックから処理を行う場合、その直前のモデルを読み込んでから学習・テストを行う
        if start_epoch != 1:
            model_file = experiment_dir + "model/cross_validation" + str(i) + "/" \
                         + "epoch" + str(start_epoch-1) + "_model.npz"
            net.load(model_file)

        # 学習データ, テストデータの準備
        train_sentences = []
        train_labels = []
        test_sentences = []
        test_labels = []

        # あらかじめ5分割しておいたデータセットを学習用とテスト用に振り分ける
        # 5/4が学習用、5/1がテスト用
        for i in range(1, 6):
            _sentence, _label = read_amazon_corpus(constants.AMAZON_BOOKDATA_DIR + "dataset" + str(j) + ".tsv")
            if k != i:
                train_sentences.extend(_sentence)
                train_labels.extend(_label)
            else:
                test_sentences.extend(_sentence)
                test_labels.extend(_label)

        # 繰り返し学習・テスト
        for epoch in range(start_epoch, end_epoch+1):
            sys.stdout.write("epoch" + str(epoch) + "...")
            sys.stdout.flush()
            net.set_train_data(train_sentences, train_labels)
            net.set_test_data(test_sentences, test_labels)
            net.train()
            net.test(experiment_dir + "out/cross_validation" + str(i) + "/epoch" + str(epoch) + ".tsv")
            net.save(experiment_dir + "model/cross_validation" + str(i) + "/epoch" + str(epoch) + "_model.npz")
            print("完了")

if __name__ == '__main__':
    # 引数パース
    parser = argparse.ArgumentParser(description='FFNN_parameter')
    parser.add_argument("--start_k", "-ks", type=int, default=1)
    parser.add_argument("--end_k", "-ke", type=int, default=5)
    parser.add_argument("--start_epoch", "-se", type=int, default=1)
    parser.add_argument("--end_epoch", "-e", type=int, default=20)
    parser.add_argument("--n_in", "-i", type=int, default=900)
    parser.add_argument("--n_mid", "-m", type=int, default=1000)
    parser.add_argument("--batchsize", "-b", type=int, default=30)
    parser.add_argument("--gpu", "-g", type=int, default=-1)
    parser.add_argument("--window_size", "-w", type=int, default=3)
    parser.add_argument("--completion", "-c", type=str, default="random")
    args = parser.parse_args()

    main(start_k=args.start_k,
         end_k=args.end_k,
         start_epoch=args.start_epoch,
         end_epoch=args.end_epoch,
         n_in=args.n_in,
         n_mid=args.n_mid,
         batchsize=args.batchsize,
         gpu=args.gpu,
         window_size=args.window_size,
         completion=args.completion)

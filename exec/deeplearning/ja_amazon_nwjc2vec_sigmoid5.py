# -*- coding: utf-8 -*-

import sys
import argparse

from jconvertor.word2vec import functions as w2v_func
from ml.deeplearning.nwjc2vec_sigmoid_5 import JW2VSigmoid5
from amazon_corpus.functions import read_amazon_corpus
import constants


def main(start_k, end_k, start_epoch, end_epoch, n_in, n_mid, batchsize, gpu, window_size):
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
    print("-------------------------------------")
    print("exec_file: ja_amazon_nwjc2vec_sigmoid5.py")
    print("start_k: " + str(start_k))
    print("end_k: " + str(end_k))
    print("start_epoch: " + str(start_epoch))
    print("end_epoch: " + str(end_epoch))
    print("入力次元数: " + str(n_in))
    print("中間次元数: " + str(n_mid))
    print("バッチサイズ: " + str(batchsize))
    print("GPU: " + str(gpu))
    print("ウィンドウサイズ: " + str(window_size))
    print("-------------------------------------")

    # 実験ディレクトリ
    experiment_dir = constants.AMAZON_DIR + "experiment/ja/sigmoid5_w2v/window" + str(window_size) + "/"

    # 実験で使用する補完関数を設定
    w2v_func.set_completion_func(w2v_func.create_random_vector)

    # 実験で使用するword2vecモデルを読み込む
    w2v_func.load_w2v(constants.W2V_MODEL_DIR + "nwjc_word_1_200_8_25_0_1e4_32_1_15.bin")

    # k_start〜k_endで5分割交差検定
    # k: k回目の検定
    for k in range(start_k, end_k+1):
        print("-------------------")
        print(str(k) + " / 5 分割目")
        print("-------------------")
        # ネットワークインスタンス作成
        net = JW2VSigmoid5(n_in, n_mid, batchsize, gpu, window_size)

        # 途中のエポックから処理を行う場合、その直前のモデルを読み込んでから学習・テストを行う
        if start_epoch != 1:
            model_file = experiment_dir + "model/cross_validation" + str(k) + "/" \
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
            _sentence, _label = read_amazon_corpus(constants.AMAZON_JP_BOOKDATA_DIR + "dataset" + str(i) + ".tsv")
            if k != i:
                train_sentences.extend(_sentence)
                train_labels.extend(_label)
            else:
                test_sentences.extend(_sentence)
                test_labels.extend(_label)

        # 繰り返し学習・テスト
        for epoch in range(start_epoch, end_epoch+1):
            sys.stdout.write("epoch" + str(epoch) + ": ")
            sys.stdout.flush()
            net.set_train_data(train_sentences, train_labels)
            net.set_test_data(test_sentences, test_labels)
            (loss1, loss2, loss3, loss4) = net.train()
            sys.stdout.write(str(loss1) + ' ' + str(loss2) + ' ' + str(loss3) + ' ' + str(loss4))
            net.test(experiment_dir + "out/cross_validation" + str(k) + "/epoch" + str(epoch) + ".tsv")
            net.save(experiment_dir + "model/cross_validation" + str(k) + "/epoch" + str(epoch) + "_model.npz")
            print("")

if __name__ == '__main__':
    # 引数パース
    parser = argparse.ArgumentParser(description='Amazonコーパスについて、sigmoid5によって5値分類する')
    parser.add_argument("--start_k", "-ks", type=int, default=1, help='5分割中、どの分割から始めるか')
    parser.add_argument("--end_k", "-ke", type=int, default=5, help='5分割中、どの分割まで行うか')
    parser.add_argument("--start_epoch", "-se", type=int, default=1, help='どのエポックから始めるか')
    parser.add_argument("--end_epoch", "-e", type=int, default=20, help='どのエポックまで行うか')
    parser.add_argument("--n_in", "-i", type=int, default=900, help='FFNNの入力次元数')
    parser.add_argument("--n_mid", "-m", type=int, default=1000, help='FFNNの中間次元数')
    parser.add_argument("--batchsize", "-b", type=int, default=30, help='学習時のバッチサイズ')
    parser.add_argument("--gpu", "-g", type=int, default=-1, help='GPUを利用するか')
    parser.add_argument("--window_size", "-w", type=int, default=3, help='フレーズとして切り取る単位')
    args = parser.parse_args()

    main(start_k=args.start_k,
         end_k=args.end_k,
         start_epoch=args.start_epoch,
         end_epoch=args.end_epoch,
         n_in=args.n_in,
         n_mid=args.n_mid,
         batchsize=args.batchsize,
         gpu=args.gpu,
         window_size=args.window_size)

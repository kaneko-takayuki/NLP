# -*- coding: utf-8 -*-

import os
import sys
import argparse

from jconvertor.word2vec import functions as w2v_func
from ml.deeplearning.nwjc2vec_sigmoid_5_softmax import NWJC2VECSigmoid5SOFTMAX
from amazon_corpus.functions import read_amazon_corpus
import constants


def main(start_k, end_k, start_epoch, end_epoch, n_in, n_mid, batchsize, gpu, window_size, patience=0):
    """
    Amazonコーパスに対して、
    sigmoidを5つ使用したモデルで、フレーズベクトルを素性として、学習・テストを行う
    :param start_k: 5分割交差検定において、どこから行うか
    :param end_k: 5分割交差検定において、どこまで行うか
    :param start_epoch: 開始エポック数
    :param end_epoch: 限界終了エポック数
    :param n_in: 入力次元数
    :param n_mid: 中間次元数
    :param batchsize: バッチサイズ
    :param gpu: GPUを利用するかどうか
    :param window_size: フレーズを区切るウィンドウサイズ
    :param patience: early stoppingに関して、様子見する回数(0の時、early stoppingはしない)
    :return: なし
    """
    print("-------------------------------------")
    print("exec_file: ja_amazon_nwjc2vec_sigmoid5_softmax.py")
    print("start_k: " + str(start_k))
    print("end_k: " + str(end_k))
    print("start_epoch: " + str(start_epoch))
    print("end_epoch: " + str(end_epoch))
    print("入力次元数: " + str(n_in))
    print("中間次元数: " + str(n_mid))
    print("バッチサイズ: " + str(batchsize))
    print("GPU: " + str(gpu))
    print("ウィンドウサイズ: " + str(window_size))
    print("patience: " + str(patience))
    print("-------------------------------------")

    # 実験ディレクトリ
    experiment_dir = constants.AMAZON_DIR + "experiment/ja/nwjc2vec_sigmoid5_softmax/window" + str(window_size) + "/"
    experiment_majority_dir = constants.AMAZON_DIR + "experiment/ja/nwjc2vec_sigmoid5_majority/window" + str(window_size) + "/"

    # 実験で使用する補完関数を設定
    w2v_func.set_completion_func(w2v_func.create_random_vector)

    # 実験で使用するword2vecモデルを読み込む
    w2v_func.load_w2v(constants.W2V_MODEL_DIR + "nwjc_word_1_200_8_25_0_1e4_32_1_15")

    # k_start〜k_endで5分割交差検定
    # k: k回目の検定
    for k in range(start_k, end_k+1):
        print(str(k) + " / 5 分割目")
        # ネットワークインスタンス作成
        net = NWJC2VECSigmoid5SOFTMAX(n_in, n_mid, batchsize, gpu, window_size)

        # 途中のエポックから処理を行う場合、その直前のモデルを読み込んでから学習・テストを行う
        if start_epoch != 1:
            model_file = experiment_dir + "model/cross_validation" + str(k) + "/" \
                         + "epoch" + str(start_epoch-1) + "_model.npz"
            net.load(model_file)

        # 学習データ, テストデータの準備
        train_sentences = []
        train_labels = []
        dev_sentences = []
        dev_labels = []
        test_sentences = []
        test_labels = []

        # あらかじめ5分割しておいたデータセットを学習用とテスト用に振り分ける
        # 3/5が学習用、1/5が検証用、5/1がテスト用
        for i in range(1, 6):
            _sentence, _label = read_amazon_corpus(constants.AMAZON_JP_BOOKDATA_DIR + "dataset" + str(i) + ".tsv")

            if ((k + i) % 5 == 0) and (patience > 0):
                dev_sentences.extend(_sentence)  # 検証用
                dev_labels.extend(_label)
            if (k + i) % 5 == 1:
                test_sentences.extend(_sentence)  # テスト用
                test_labels.extend(_label)
            else:
                train_sentences.extend(_sentence)  # 学習用
                train_labels.extend(_label)

        # データのセット
        net.set_train_data(train_sentences, train_labels)
        net.set_dev_data(dev_sentences, dev_labels)
        net.set_test_data(test_sentences, test_labels)

        # 繰り返し学習・テスト
        print("--------------------------------------------------------------")
        print(" epoch | train_loss | test_accuracy | dev_accuracy ")
        for epoch in range(start_epoch, end_epoch+1):
            sys.stdout.write(str(epoch).center(7) + '|')
            sys.stdout.flush()

            # 学習フェーズ
            train_loss = '-'
            if os.path.exists(experiment_majority_dir + "model/cross_validation" + str(k) + "/epoch" + str(epoch) + "_model.npz"):
                net.load(experiment_majority_dir + "model/cross_validation" + str(k) + "/epoch" + str(epoch) + "_model.npz")
            else:
                train_loss = net.train()
            sys.stdout.write(str(train_loss)[:10].center(12) + '|')
            sys.stdout.flush()

            # テストフェーズ
            test_accuracy = net.test(experiment_dir + "out/cross_validation" + str(k) + "/epoch" + str(epoch) + ".tsv")
            sys.stdout.write(str(test_accuracy)[:13].center(15) + '|')
            sys.stdout.flush()

            # モデルの保存
            net.save(experiment_dir + "model/cross_validation" + str(k) + "/epoch" + str(epoch) + "_model.npz")

            # 収束検証フェーズ
            if patience > 0:
                dev_accuracy, early_stopping_flag = net.dev(patience)
                sys.stdout.write(str(dev_accuracy)[:12].center(14))
                sys.stdout.flush()
                if early_stopping_flag:
                    break
            print()
        print("--------------------------------------------------------------")

if __name__ == '__main__':
    # 引数パース
    parser = argparse.ArgumentParser(description='ja_Amazonコーパスについて、nwjc2vecとsigmoid5_softmaxによって5値分類する')
    parser.add_argument("--start_k", "-ks", type=int, default=1, help='5分割中、どの分割から始めるか')
    parser.add_argument("--end_k", "-ke", type=int, default=5, help='5分割中、どの分割まで行うか')
    parser.add_argument("--start_epoch", "-se", type=int, default=1, help='どのエポックから始めるか')
    parser.add_argument("--end_epoch", "-e", type=int, default=20, help='どのエポックまで行うか')
    parser.add_argument("--n_in", "-i", type=int, default=900, help='FFNNの入力次元数')
    parser.add_argument("--n_mid", "-m", type=int, default=1000, help='FFNNの中間次元数')
    parser.add_argument("--batchsize", "-b", type=int, default=30, help='学習時のバッチサイズ')
    parser.add_argument("--gpu", "-g", type=int, default=-1, help='GPUを利用するか')
    parser.add_argument("--window_size", "-w", type=int, default=3, help='フレーズとして切り取る単位')
    parser.add_argument("--patience", "-p", type=int, default=5, help='early stopping判定の時の様子見の回数')
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
         patience=args.patience)

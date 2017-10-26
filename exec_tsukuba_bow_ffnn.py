# -*- coding: utf-8 -*-

import os
import sys

from ml.deeplearning.j_bow_ffnn import JBOWFFNN
from tsukuba_corpus.functions import read_tsukuba_corpus
import constants


def main(n_in, n_mid, n_out, batchsize, gpu, n_epoch):
    # cudaへのパス
    os.environ["PATH"] = "/usr/local/cuda-7.5/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

    # 実験ディレクトリ
    experiment_dir = "tsukuba_corpus/Test_FFNN_W2V/zero/"

    # 5分割交差検定
    for i in range(1, 6):
        print("-------------------")
        print(str(i) + " / 5 分割目")
        print("-------------------")
        # ネットワークインスタンス作成
        net = JBOWFFNN(n_in, n_mid, n_out, batchsize, gpu)

        # 学習データ, テストデータの準備
        train_sentences = []
        train_labels = []
        test_sentences = []
        test_labels = []

        for j in range(1, 6):
            _sentence, _label = read_tsukuba_corpus(constants.TSUKUBA_DATA_DIR + "dataset" + str(j) + ".tsv")
            # 5分割したデータのうち、一つだけをテストデータに回し、それ以外を学習データとする
            if i != j:
                train_sentences.extend(_sentence)
                train_labels.extend(_label)
            else:
                test_sentences.extend(_sentence)
                test_labels.extend(_label)

        # 繰り返し学習・テスト
        for epoch in range(1, n_epoch+1):
            sys.stdout.write("epoch" + str(epoch) + "...")
            sys.stdout.flush()
            net.set_train_data(train_sentences, train_labels)
            net.set_test_data(test_sentences, test_labels)
            net.train()
            net.test(experiment_dir + "out/cross_validation" + str(i) + "/epoch" + str(epoch) + ".tsv")
            net.save(experiment_dir + "model/cross_validation" + str(i) + "/epoch" + str(epoch) + "_model.npz")
            print("完了")

if __name__ == '__main__':
    main(900, 1000, 2, 20, 0, 3)

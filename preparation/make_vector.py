# -*- coding: utf-8 -*-

import os
import sys

from econvertor.word2vec import functions as w2v_func
from amazon_corpus.functions import read_amazon_corpus
from ml.deeplearning import make_sv_lstm
import constants


def main():
    """
    5分割交差検定1回目限定スクリプト
    :return: 
    """
    # cudaへのパス
    os.environ["PATH"] = "/usr/local/cuda-7.5/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

    # 実験ディレクトリ
    base = os.path.dirname(os.path.abspath(__file__))
    experiment_dir = base + "/../amazon_corpus/lstm_w2v/random/"

    # 実験で使用する補完関数を設定
    w2v_func.set_completion_func(w2v_func.create_random_vector)

    # 実験で使用するword2vecモデルを読み込む
    w2v_func.load_w2v("/home/kaneko-takayuki/NLP/w2v_model/GoogleNews-vectors-negative300.bin")

    for epoch in range(1, 51):
        sys.stdout.write("epoch" + str(epoch) + "...")
        sys.stdout.flush()
        net = make_sv_lstm.MakeSentenceVectorLSTM(n_in=300, n_mid=1000, n_out=5, batchsize=30, gpu=0)
        net.load(experiment_dir + "model_2layer/cross_validation1/epoch" + str(epoch) + "_model.npz")

        # テストデータについて
        _sentence, _label = read_amazon_corpus(constants.AMAZON_BOOKDATA_DIR + "dataset1.tsv")
        net.set_test_data(_sentence, _label)
        net.output(experiment_dir + "out/cross_validation1/epoch" + str(epoch) + "_test.tsv")
        # 学習データについて
        for i in range(2, 6):
            _sentence, _label = read_amazon_corpus(constants.AMAZON_BOOKDATA_DIR + "dataset" + str(i) + ".tsv")
            net.set_test_data(_sentence, _label)
            net.output(experiment_dir + "out/cross_validation1/epoch" + str(epoch) + "_train" + str(i-1) + ".tsv")
        print("完了")

if __name__ == '__main__':
    main()

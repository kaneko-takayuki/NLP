# -*- coding: utf-8 -*-

import sys
import argparse

from jconvertor.word2vec import functions as jw2v_func
from econvertor.word2vec import functions as ew2v_func
from ja_eng_lib.functions import read_all

from ml.deeplearning.ja_to_eng import JAtoENG


def main(start_epoch, end_epoch, n_in, n_mid, n_out, batchsize, gpu):
    # ネットワークインスタンス作成
    net = JAtoENG(n_in, n_mid, n_out, batchsize, gpu)

    # 途中のエポックから処理を行う場合、その直前のモデルを読み込んでから学習・テストを行う
    if start_epoch != 1:
        model_file = "/home/kaneko-takayuki/NLP/ja_eng_lib/model_mse/epoch" + str(start_epoch) + "_model.npz"
        net.load(model_file)

    # 実験で使用する補完関数を設定
    jw2v_func.set_completion_func(jw2v_func.return_none)
    ew2v_func.set_completion_func(ew2v_func.return_none)

    # 実験で使用するword2vecモデルを読み込む
    jw2v_func.load_w2v("/home/kaneko-takayuki/NLP/w2v_model/nwcj_word_1_200_8_25_0_1e4_32_1_15.bin")
    ew2v_func.load_w2v("/home/kaneko-takayuki/NLP/w2v_model/GoogleNews-vectors-negative300.bin")

    net.set_train_data(read_all())

    # 繰り返し学習・テスト
    for epoch in range(start_epoch, end_epoch + 1):
        sys.stdout.write("epoch" + str(epoch) + ": ")
        sys.stdout.flush()
        net.train()
        net.save("/home/kaneko-takayuki/NLP/ja_eng_lib/model_mse/epoch" + str(epoch) + "_model.npz")

if __name__ == '__main__':
    # 引数パース
    parser = argparse.ArgumentParser(description='日本単語ベクトルから英語単語ベクトルへ変換するモデルを作成')
    parser.add_argument("--start_epoch", "-se", type=int, default=1, help='どのエポックから始めるか')
    parser.add_argument("--end_epoch", "-e", type=int, default=50, help='どのエポックまで行うか')
    parser.add_argument("--n_in", "-i", type=int, default=200, help='FFNNの入力次元数')
    parser.add_argument("--n_mid", "-m", type=int, default=1000, help='FFNNの中間次元数')
    parser.add_argument("--n_out", "-o", type=int, default=300, help='FFNNの出力次元数')
    parser.add_argument("--batchsize", "-b", type=int, default=30, help='学習時のバッチサイズ')
    parser.add_argument("--gpu", "-g", type=int, default=-1, help='GPUを利用するか')
    args = parser.parse_args()

    main(start_epoch=args.start_epoch,
         end_epoch=args.end_epoch,
         n_in=args.n_in,
         n_mid=args.n_mid,
         n_out=args.n_out,
         batchsize=args.batchsize,
         gpu=args.gpu)

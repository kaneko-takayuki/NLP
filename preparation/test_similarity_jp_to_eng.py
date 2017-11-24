# -*- coding: utf-8 -*-

import os
import sys
import argparse

from jconvertor.word2vec import functions as jw2v_func
from econvertor.word2vec import functions as ew2v_func
from ja_eng_lib.functions import read_all
from ml.deeplearning.ja_to_eng import JAtoENG


def main(n_in, n_mid, n_out, gpu, dir_name):
    # cudaへのパス
    os.environ["PATH"] = "/usr/local/cuda-7.5/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

    # ネットワークインスタンス作成
    net = JAtoENG(n_in, n_mid, n_out, None, gpu)
    net.load(dir_name + "epoch200_model.npz")

    # 実験で使用する補完関数を設定
    jw2v_func.set_completion_func(jw2v_func.return_none)
    ew2v_func.set_completion_func(ew2v_func.return_none)

    # 実験で使用するword2vecモデルを読み込む
    jw2v_func.load_w2v("/home/kaneko-takayuki/NLP/w2v_model/nwcj_word_1_200_8_25_0_1e4_32_1_15.bin")
    ew2v_func.load_w2v("/home/kaneko-takayuki/NLP/w2v_model/GoogleNews-vectors-negative300.bin")

    num_words = sum(1 for _ in open("/home/kaneko-takayuki/NLP/ja_eng_lib/similarity_test/test_review_words_corpus.txt"))
    n = 1
    # 繰り返し入力を得て、マッチする英単語を求める
    with open("/home/kaneko-takayuki/NLP/ja_eng_lib/similarity_test/test_review_words_corpus.txt", 'r') as i:
        with open("/home/kaneko-takayuki/NLP/ja_eng_lib/similarity_test/method2/test_review_words_result.txt", 'w') as o:
            for j_word in i:
                sys.stdout.write("\r" + str(n) + " / " + str(num_words))
                sys.stdout.flush()
                n += 1
                j_word = j_word.replace('\n', '')
                matched_eword = net.most_similar(j_word)
                if len(matched_eword) == 0:
                    continue  # KeyErrorでcosine類似度が取れなかった
                o.write(j_word + ": " + str(matched_eword) + '\n')

if __name__ == '__main__':
    # 引数パース
    parser = argparse.ArgumentParser(description='日本単語ベクトルから英語単語ベクトルへ変換するモデルを作成')
    parser.add_argument("--n_in", "-i", type=int, default=200, help='FFNNの入力次元数')
    parser.add_argument("--n_mid", "-m", type=int, default=1000, help='FFNNの中間次元数')
    parser.add_argument("--n_out", "-o", type=int, default=300, help='FFNNの出力次元数')
    parser.add_argument("--gpu", "-g", type=int, default=-1, help='GPUを利用するか')
    parser.add_argument("--dir", "-d", type=str, default="", help='モデルファイルが格納されているディレクトリ')
    args = parser.parse_args()

    main(args.n_in, args.n_mid, args.n_out, args.gpu, args.dir)

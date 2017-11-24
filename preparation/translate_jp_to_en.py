# -*- coding: utf-8 -*-

import os
import sys
import argparse

from jconvertor import spliter
from jconvertor.word2vec import functions as jw2v_func
from ml.deeplearning.ja_to_eng import JAtoENG


def main(n_in, n_mid, n_out, gpu, dir_name):
    # cudaへのパス
    os.environ["PATH"] = "/usr/local/cuda-7.5/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

    # ネットワークインスタンス作成
    net = JAtoENG(n_in, n_mid, n_out, None, gpu)
    net.load(dir_name + "epoch230_model.npz")

    # 実験で使用する補完関数を設定
    jw2v_func.set_completion_func(jw2v_func.return_none)

    # 実験で使用するword2vecモデルを読み込む
    jw2v_func.load_w2v("/home/kaneko-takayuki/NLP/w2v_model/nwcj_word_1_200_8_25_0_1e4_32_1_15.bin")

    n = 1
    keys = []
    # 繰り返し入力を得て、マッチする英単語を求める
    with open("/home/kaneko-takayuki/NLP/amazon_corpus/data/jp/books/all.tsv", 'r') as i:
        print("単語を読み取り中...")
        for line in i:
            if n % 10 == 0:
                sys.stdout.write("\r" + str(n) + " / 4000")
            sys.stdout.flush()
            n += 1
            items = line.replace('\n', '').split('\t')
            words = spliter.words(items[1])  # 単語リスト

            for word in words:
                if word in keys:
                    continue  # 既に登録済

                jp_word_vector = net.foward(word)  # 日単語の英ベクトル
                if len(jp_word_vector) == 0:
                    continue

                with open("/home/kaneko-takayuki/NLP/ja_eng_lib/dictionary.txt", 'a') as o:
                    keys.append(word)
                    vec_str = ','.join([str(v) for v in jp_word_vector])
                    o.write(word + '\t' + vec_str + '\n')


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
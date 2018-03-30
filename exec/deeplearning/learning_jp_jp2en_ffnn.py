# -*- coding: utf-8 -*-

import argparse
import functools
from myio.phase1 import read
from dto.conf import ConfFFNN
from dto.conf import write_confFFNN
from deeplearning.jp2en import train
from deeplearning.generator import generate_with_slice
from deeplearning.generator import random_vector


def main(train_files: list,
         test_files: list,
         model_directory: str,
         windowsize: int,
         start: int=1,
         end: int=10,
         n_in: int=300,
         n_mid: int=100,
         n_out: int=5,
         batchsize: int=10,
         gpu: int=-1):

    # 設定の表示
    print('train_files: ' + ', '.join(train_files))
    print('test_files: ' + ', '.join(test_files))
    print('model_directory: ' + model_directory)
    print('windowsize: ' + str(windowsize))
    print('start: ' + str(start))
    print('end: ' + str(end))
    print('n_in: ' + str(n_in))
    print('n_mid: ' + str(n_mid))
    print('n_out: ' + str(n_out))
    print('batchsize: ' + str(batchsize))
    print('gpu: ' + str(gpu))

    # 学習の設定関連
    conf = ConfFFNN(n_in, n_mid, n_out, batchsize, gpu)

    # ジェネレータ
    generator = functools.partial(generate_with_slice, windowsize, random_vector(300, 0, 0))

    # startエポック〜endエポックまで学習を行う
    for e in range(start, end + 1):
        print("epoch: " + str(e))

        # 学習する
        for train_file in train_files:
            print('train_file: ' + train_file)
            conf = train(conf=conf, learning_data=read(train_file), generator=generator)

        # 保存する
        save_file = model_directory + "epoch" + str(e)
        write_confFFNN(save_file, conf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='日->英ベクトルを用いて、日本語コーパスについて学習を行う')
    parser.add_argument("--learning", "-l", type=str, default=[], nargs='*', help='学習コーパスファイル(複数指定可能)')
    parser.add_argument("--test", "-t", type=str, default=[], nargs='*', help='テストコーパスファイル(複数指定可能)')
    parser.add_argument("--windowsize", "-w", type=int, help='フレーズを切り取る単位')
    parser.add_argument("--model_directory", "-d", type=str, help='モデルを保存するディレクトリ')
    parser.add_argument("--start", "-s", type=int, default=1, help='何エポック目から始めるか')
    parser.add_argument("--end", "-e", type=int, default=50, help='何エポック目まで学習を行うか')
    parser.add_argument("--n_in", "-i", type=int, default=200, help='モデルの入力次元数')
    parser.add_argument("--n_mid", "-m", type=int, default=1000, help='モデルの中間次元数')
    parser.add_argument("--n_out", "-o", type=int, default=300, help='モデルの出力次元数')
    parser.add_argument("--batchsize", "-b", type=int, default=30, help='学習時のバッチサイズ')
    parser.add_argument("--gpu", "-g", type=int, default=-1, help='GPUを利用するか')
    args = parser.parse_args()

    main(args.learning, args.test, args.model_directory, args.windowsize, args.start, args.end, args.n_in, args.n_mid, args.n_out, args.batchsize, args.gpu)

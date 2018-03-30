# -*- coding: utf-8 -*-

import numpy as np
import chainer
from chainer import Variable
from dto.conf import ConfFFNN


def init_conf(n_in, n_mid, n_out, batchsize, gpu=-1):
    """
    FFNNconfインスタンスを生成する
    :param n_in: 入力次元数
    :param n_mid: 中間次元数
    :param n_out: 出力次元数
    :param batchsize: バッチサイズ
    :param gpu: GPUのID
    :return: confインスタンス
    """
    return ConfFFNN(n_in, n_mid, n_out, batchsize, gpu)


def train(conf: ConfFFNN, learning_data: list, generator):
    """
    1エポック分学習させる
    :param conf: FFNNの設定ファイル
    :param learning_data: 学習データlist(Data)
    :param generator: Data -> list()
    :return: なし
    """
    # ランダム配列作成し、それに沿って学習データリストを並び替え
    n = len(learning_data)
    perm = iter(np.random.permutation(n))
    sorted_learning = [learning_data[i] for i in perm]

    sum_loss = 0

    # batchsize個ずつ抜き出し学習
    for i in range(0, n - conf.batchsize, conf.batchsize):
        # データを切り出し、実際に学習するベクトルを生成する
        # generator(list(Phrase)) -> list(LearningData))
        generated_data = []
        for data in sorted_learning[i:(i + conf.batchsize)]:
            generated_data.extend(generator(data))

        # ジェネレートされたデータを学習できる状態に持っていく(Variable化)
        xp_train_labels = conf.xp.asarray(list(map(lambda x: x.label, generated_data))).astype(conf.xp.int32)
        xp_train_inputs = conf.xp.asarray(list(map(lambda x: x.vector, generated_data))).astype(conf.xp.float32)
        variable_train_labels = Variable(xp_train_labels)
        variable_train_inputs = Variable(xp_train_inputs)

        # 学習
        with chainer.using_config('train', True):
            conf.model.cleargrads()
            loss = conf.model(variable_train_inputs, variable_train_labels)
            sum_loss += loss.data
            loss.backward()
            conf.optimizer.update()

    print('loss: ' + str(sum_loss))

    return conf

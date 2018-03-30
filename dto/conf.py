# -*- coding: utf-8 -*-

import numpy as np
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from deeplearning.model.ffnn import FFNN


class ConfFFNN:
    def __init__(self, n_in: int, n_mid: int, n_out: int, batchsize: int, gpu: int=-1):
        # 基本設定
        self.n_in = n_in
        self.n_mid = n_mid
        self.n_out = n_out
        self.batchsize = batchsize
        self.gpu = gpu

        # モデル
        self.model = FFNN(n_in, n_mid, n_out)

        # GPU関連設定
        self.xp = np if gpu < 0 else cuda.cupy
        if gpu >= 0:
            cuda.get_device_from_id(gpu).use()
            cuda.check_cuda_available()
            self.model.to_gpu()

        # 最適化手法(Adam)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)


def write_confFFNN(file_name: str, conf):
    """
    モデルを保存する
    :param file_name: 保存するファイル名
    :param conf: ディープラーニング設定ファイル
    :return: なし
    """
    # CPUモードで統一的に保存する
    conf.model.to_cpu()
    serializers.save_npz(file_name, conf.model)

    # GPU設定
    if conf.gpu >= 0:
        cuda.get_device_from_id(conf.gpu).use()
        cuda.check_cuda_available()
        conf.model.to_gpu()

# -*- coding: utf-8 -*-

from tmp.ml import ConfFFNN


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


def train(conf, learning_data):
    """
    1エポック分学習させる
    :param conf: FFNNの設定ファイル
    :param learning_data: 学習データ
    :return: なし
    """
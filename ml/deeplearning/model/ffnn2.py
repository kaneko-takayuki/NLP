# -*- coding: utf-8 -*-

import chainer
import chainer.links as L
import chainer.functions as F


class FFNN(chainer.Chain):
    """
    入力次元数n_in, 中間ノード数n_mid, 出力次元数n_out
    の3層から成るフィードフォワードニューラルネットワーク
    """
    def __init__(self, n_in, n_mid, n_out):
        super(FFNN, self).__init__(
            l1=L.Linear(n_in, n_mid),
            l2=L.Linear(n_mid, n_mid),
            l3=L.Linear(n_mid, n_out),
        )

    def __call__(self, x, y):
        """
        誤差関数
        :param x: 入力ベクトル
        :param y: 教師ラベル
        :return: 誤差
        """
        return F.mean_squared_error(self.fwd(x), y)

    def fwd(self, x):
        """
        フォワード処理
        :param x: 入力ベクトル
        :return: フォワード処理結果
        """
        h1 = self.l1(x)
        h2 = self.l2(h1)
        return self.l3(h2)

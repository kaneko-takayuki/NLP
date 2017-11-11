# -*- coding: utf-8 -*-

import chainer
import chainer.links as L
import chainer.functions as F


class LSTM(chainer.Chain):
    """
    入力ノード次元n_in、中間ノード次元n_units、出力ノード次元n_out
    の3層から成るLSTMネットワーク
    """

    def __init__(self, n_in, n_units, n_out):
        super(LSTM, self).__init__(
            l1=L.LSTM(n_in, n_units),
            l2=L.Linear(n_units, n_out),
        )

    def __call__(self, x, y):
        return F.softmax_cross_entropy(self.fwd(x), y)

    def fwd(self, x):
        h0 = F.dropout(x)
        h1 = F.dropout(self.l1(h0))
        return self.l2(h1)

    def get_compression_vector(self, x):
        h0 = F.dropout(x)
        return self.l1(h0)

    def reset_state(self):
        self.l1.reset_state()

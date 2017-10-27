# -*- coding: utf-8 -*-

import chainer
import chainer.links as L
import chainer.functions as F


class SIGMOID5(chainer.Chain):
    """
    入力次元数n_in, 中間次元数n_mid, 出力層は4つに分岐してsigmoidを掛ける
    それらを閾値と見なして、分類を行う
    """
    def __init__(self, n_in, n_mid):
        super(SIGMOID5, self).__init__(
            l1=L.Linear(n_in, n_mid),
            l2_1=L.Linear(n_mid, 1),
            l2_2=L.Linear(n_mid, 1),
            l2_3=L.Linear(n_mid, 1),
            l2_4=L.Linear(n_mid, 1),
        )

    def loss1(self, x, y):
        return F.mean_squared_error(self.fwd1(x), y)

    def loss2(self, x, y):
        return F.mean_squared_error(self.fwd2(x), y)

    def loss3(self, x, y):
        return F.mean_squared_error(self.fwd3(x), y)

    def loss4(self, x, y):
        return F.mean_squared_error(self.fwd4(x), y)

    def fwd1(self, x):
        h1 = F.dropout(F.relu(self.l1(x)))
        return F.sigmoid(self.l2_1(h1))

    def fwd2(self, x):
        h1 = F.dropout(F.relu(self.l1(x)))
        return F.sigmoid(self.l2_2(h1))

    def fwd3(self, x):
        h1 = F.dropout(F.relu(self.l1(x)))
        return F.sigmoid(self.l2_3(h1))

    def fwd4(self, x):
        h1 = F.dropout(F.relu(self.l1(x)))
        return F.sigmoid(self.l2_4(h1))

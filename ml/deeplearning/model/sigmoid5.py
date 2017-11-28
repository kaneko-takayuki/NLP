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
            l2=L.Linear(n_mid, n_mid),
            l3_1=L.Linear(n_mid, 1),  # ラベルが1以上かどうかを判別するノード
            l3_2=L.Linear(n_mid, 1),  # ラベルが2以上かどうかを判別するノード
            l3_3=L.Linear(n_mid, 1),  # ラベルが3以上かどうかを判別するノード
            l3_4=L.Linear(n_mid, 1),  # ラベルが4以上かどうかを判別するノード
        )

    # 損失関数4種類
    def loss1(self, x, y):
        return F.mean_squared_error(self.fwd1(x), y)

    def loss2(self, x, y):
        return F.mean_squared_error(self.fwd2(x), y)

    def loss3(self, x, y):
        return F.mean_squared_error(self.fwd3(x), y)

    def loss4(self, x, y):
        return F.mean_squared_error(self.fwd4(x), y)

    # フォワード処理4種類
    def fwd1(self, x):
        h1 = F.dropout(F.relu(self.l1(x)))
        h2 = F.dropout(F.relu(self.l2(h1)))
        return F.sigmoid(self.l3_1(h2))

    def fwd2(self, x):
        h1 = F.dropout(F.relu(self.l1(x)))
        h2 = F.dropout(F.relu(self.l2(h1)))
        return F.sigmoid(self.l3_2(h2))

    def fwd3(self, x):
        h1 = F.dropout(F.relu(self.l1(x)))
        h2 = F.dropout(F.relu(self.l2(h1)))
        return F.sigmoid(self.l3_3(h2))

    def fwd4(self, x):
        h1 = F.dropout(F.relu(self.l1(x)))
        h2 = F.dropout(F.relu(self.l2(h1)))
        return F.sigmoid(self.l3_4(h2))

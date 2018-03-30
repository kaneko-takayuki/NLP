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
    # 第2層の出力値を受け取って、出力値と誤差を返す
    def loss1(self, h2, y):
        output = self.fwd1(h2)
        return output, F.mean_squared_error(output, y)

    def loss2(self, h2, y):
        output = self.fwd2(h2)
        return output, F.mean_squared_error(output, y)

    def loss3(self, h2, y):
        output = self.fwd3(h2)
        return output, F.mean_squared_error(output, y)

    def loss4(self, h2, y):
        output = self.fwd4(h2)
        return output, F.mean_squared_error(output, y)

    # 第2層までのフォワード処理
    def fwd_until_l2(self, x):
        h1 = F.dropout(F.relu(self.l1(x)))
        return F.dropout(F.relu(self.l2(h1)))

    # 第3層のフォワード処理
    def fwd1(self, h2):
        return F.sigmoid(self.l3_1(h2))

    def fwd2(self, h2):
        return F.sigmoid(self.l3_2(h2))

    def fwd3(self, h2):
        return F.sigmoid(self.l3_3(h2))

    def fwd4(self, h2):
        return F.sigmoid(self.l3_4(h2))

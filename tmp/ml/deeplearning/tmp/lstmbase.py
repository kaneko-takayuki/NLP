# -*- coding: utf-8 -*-

from abc import abstractmethod

import chainer
import chainer.functions as F
import numpy as np
import six
from chainer import cuda
from chainer import serializers

from tmp.ml import MLBases


class LSTMBases(MLBases):
    def __init__(self, batchsize, gpu=-1):
        MLBases.__init__(self)

        # DeepLearningの設定関連
        self.model = None
        self.optimizer = None
        self.xp = np if gpu < 0 else cuda.cupy
        self.batchsize = batchsize
        self.gpu = gpu

    @abstractmethod
    def convert(self, sentence, label):
        pass

    @abstractmethod
    def output(self, file_name, sentence, corr_label, pred_labels):
        pass

    def train(self):
        """
        1エポック分の学習を行う
        :return: なし
        """
        # ランダム配列作成
        perm = np.random.permutation(self.num_train_data())

        # batchsize個ずつデータを入れて学習させていく
        for i in six.moves.range(0, self.num_train_data() - self.batchsize, self.batchsize):
            sum_loss = 0  # 蓄積誤差
            # 実際に学習させるデータとラベル
            train_inputs = []
            train_labels = []

            # ランダムにbatchsize個のデータを取り出し、convertで設定した方法で学習データを作り出す
            for j in six.moves.range(i, i + self.batchsize):
                j_input, j_label = self.convert(self.train_sentences[perm[j]], self.train_labels[perm[j]])
                train_inputs.append(j_input)
                train_labels.append(j_label)

            # 学習処理
            self.model.cleargrads()  # 勾配の初期化
            self.model.reset_state()  # 内部状態の初期化

            # 学習データについて順番に見ていく
            for train_input, train_label in zip(train_inputs, train_labels):
                word_len = len(train_input)
                # j: 何番目の単語について見る
                for j in range(word_len):
                    x = chainer.Variable(self.xp.asarray([train_input[j]]).astype(np.float32))
                    t = chainer.Variable(self.xp.asarray(train_label).astype(np.int32))

                    # 最後の出力に対してのみ誤差を計算し、逆伝播
                    if j == word_len - 1:
                        with chainer.using_config('train', True):
                            sum_loss += self.model(x, t)
                    else:
                        with chainer.using_config('train', False):
                            self.model.fwd(x)
            # 蓄積した誤差を逆伝播
            sum_loss.backward()
            self.optimizer.update()

    def test(self, file_name):
        """
        1エポック分のテストを行う
        :param file_name: 出力ファイル名
        :return: なし
        """
        # 1つずつデータを取り出し、テストを行う
        for i in six.moves.range(self.num_test_data()):
            # テストを行うデータ
            i_input, i_label = self.convert(self.test_sentences[i], self.test_labels[i])
            i_input = np.asarray(i_input).astype(np.float32)
            i_input = chainer.Variable(self.xp.asarray(i_input))

            # ラベルの予測
            with chainer.using_config('train', False):
                pred_labels = F.softmax(self.model.fwd(i_input))

            # 出力
            self.output(file_name, self.test_sentences[i], self.test_labels[i], pred_labels)

    def save(self, file_name):
        """
        モデルを保存する
        :param file_name: セーブファイル名
        :return: なし
        """
        # CPUモードで保存
        self.model.to_cpu()
        serializers.save_npz(file_name, self.model)

        # GPU設定
        if self.gpu >= 0:
            chainer.cuda.get_device_from_id(self.gpu).use()
            cuda.check_cuda_available()
            self.model.to_gpu()

    def load(self, file_name):
        """
        モデルを読み込む
        :param file_name: ロードファイル名
        :return: なし
        """
        # モデルをロードする
        serializers.load_npz(file_name, self.model)

        # GPU設定
        if self.gpu >= 0:
            chainer.cuda.get_device_from_id(self.gpu).use()
            cuda.check_cuda_available()
            self.model.to_gpu()

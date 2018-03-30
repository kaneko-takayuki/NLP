# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import numpy as np
import six
from chainer import cuda
from chainer import serializers

from tmp.ml import MLBases


class DLBases(MLBases):
    def __init__(self, batchsize, gpu=-1):
        MLBases.__init__(self)

        # DeepLearningの設定関連
        self.model = None
        self.optimizer = None
        self.xp = np if gpu < 0 else cuda.cupy
        self.batchsize = batchsize
        self.gpu = gpu

        # 検証用データ
        self.dev_sentences = []
        self.dev_labels = []

        # early stoppingの為に、検証用データを保持しておくリスト
        self.dev_accuracy = []

    def set_dev_data(self, ts, tl):
        """
        検証用データをセット
        :param ts: 検証文章リスト <class: 'list'>
        :param tl: 検証ラベルリスト <class: 'list'>
        :return: なし
        """
        self.dev_sentences = ts
        self.dev_labels = tl

    def num_dev_data(self):
        """
        検証データ数を返す
        :return: 学習データ数
        """
        return len(self.dev_sentences)

    def train(self):
        """
        1エポック分の学習を行う
        :return: なし
        """
        # ランダム配列作成
        perm = np.random.permutation(self.num_train_data())

        # batchsize個ずつデータを入れて学習させていく
        for i in six.moves.range(0, self.num_train_data() - self.batchsize, self.batchsize):
            # 実際に学習させるデータとラベル
            train_inputs = []
            train_labels = []

            # ランダムにbatchsize個のデータを取り出し、convertで設定した方法で学習データを作り出す
            for j in six.moves.range(i, i + self.batchsize):
                j_input, j_label = self.convert(self.train_sentences[perm[j]], self.train_labels[perm[j]])
                train_inputs.extend(j_input)
                train_labels.extend(j_label)

            # リストからnumpy化
            train_inputs = np.asarray(train_inputs).astype(np.float32)
            train_labels = np.asarray(train_labels).astype(np.int32)

            # 学習処理
            train_inputs = chainer.Variable(self.xp.asarray(train_inputs))
            train_labels = chainer.Variable(self.xp.asarray(train_labels))
            with chainer.using_config('train', True):
                self.model.cleargrads()
                loss = self.model(train_inputs, train_labels)
                loss.backward()
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
            self.model.model_to_gpu()

    def early_stopping(self, patience):
        """
        early stoppingによって、学習を打ち切るかどうか判定する
        :param patience: 様子見の回数
        :return: 打ち切る場合True, 打ち切らない場合False
        """
        # early stopping判定するのに、十分な回数行われていない
        if len(self.dev_accuracy) < patience + 1:
            return False

        # 直近patience回の間に、検証用データで最大精度が出ていれば、学習を続ける
        threshold_accuracy = self.dev_accuracy[-(patience + 1)]
        for accuracy in self.dev_accuracy[-patience:]:
            if threshold_accuracy < accuracy:
                return False

        return True

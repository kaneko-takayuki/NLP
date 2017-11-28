# -*- coding: utf-8 -*-

import copy
import numpy as np
import six
import chainer
from chainer import optimizers
from chainer import serializers
from chainer import cuda

from ml.base import MLBases
from ml.deeplearning.model import sigmoid5
from jconvertor import spliter
from jconvertor import vectorizer


def threshold(labels, x):
    """
    labelsのそれぞれに対して、xより大きいものを1、それ以外を0となるように変換する
    :param labels: ラベルリスト
    :param x: 閾値
    :return: 変換済ラベルリスト
    """
    _labels = copy.deepcopy(labels)
    for i in range(len(_labels)):
        if _labels[i] <= x:
            _labels[i] = 0.0
        else:
            _labels[i] = 1.0
    return _labels


class JW2VSigmoid5(MLBases):
    def __init__(self, n_in, n_mid, batchsize, gpu=-1, window_size=1):
        MLBases.__init__(self)

        # モデル構築
        self.model = sigmoid5.SIGMOID5(n_in, n_mid)

        # GPU設定
        if gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            cuda.check_cuda_available()
            self.model.to_gpu()

        # 最適化手法をAdamに設定
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

        # パラメータ保持
        self.xp = np if gpu < 0 else cuda.cupy
        self.batchsize = batchsize
        self.gpu = gpu
        self.window_size = window_size

    def train(self):
        """
        1エポック分の学習を行う
        :return: 各ノード(1〜4番目)の誤差和
        """
        # ランダム配列作成
        perm = np.random.permutation(self.num_train_data())

        # 誤差の総和
        sum_loss1 = 0.0
        sum_loss2 = 0.0
        sum_loss3 = 0.0
        sum_loss4 = 0.0

        # batchsize個ずつのミニバッチを作成し、その単位で学習を行う
        for i in six.moves.range(0, self.num_train_data() - self.batchsize, self.batchsize):
            # 実際に学習させるデータとラベル
            train_inputs = []
            train_labels = []

            # ランダムにbatchsize個のデータを取り出し、convertで設定した方法で学習データを作り出す
            for j in six.moves.range(i, i + self.batchsize):
                j_input, j_label = self.convert(self.train_sentences[perm[j]], self.train_labels[perm[j]])
                train_inputs.extend(j_input)
                train_labels.extend(j_label)

            # 学習データの型を学習用に変更
            train_inputs = np.asarray(train_inputs).astype(np.float32)
            variable_train_inputs = chainer.Variable(self.xp.asarray(train_inputs))
            train_labels = np.asarray(train_labels).astype(np.float32)

            # 学習処理
            with chainer.using_config('train', True):
                self.model.cleargrads()

                # 第2層までの出力値を保持しておく
                value_until_l2 = self.model.fwd_until_l2(variable_train_inputs)

                # ラベルが1以上かどうかの学習
                variable_train_labels = chainer.Variable(self.xp.asarray(threshold(train_labels, 0)))
                loss1 = self.model.loss1(value_until_l2, variable_train_labels)
                sum_loss1 += loss1.data
                loss1.backward()

                # ラベルが2以上かどうかの学習
                variable_train_labels = chainer.Variable(self.xp.asarray(threshold(train_labels, 1)))
                loss2 = self.model.loss2(value_until_l2, variable_train_labels)
                sum_loss2 += loss2.data
                loss2.backward()

                # ラベルが3以上かどうかの学習
                variable_train_labels = chainer.Variable(self.xp.asarray(threshold(train_labels, 2)))
                loss3 = self.model.loss3(value_until_l2, variable_train_labels)
                sum_loss3 += loss3.data
                loss3.backward()

                # ラベルが4以上かどうかの学習
                variable_train_labels = chainer.Variable(self.xp.asarray(threshold(train_labels, 3)))
                loss4 = self.model.loss4(value_until_l2, variable_train_labels)
                sum_loss4 += loss4.data
                loss4.backward()
                self.optimizer.update()

        return sum_loss1, sum_loss2, sum_loss3, sum_loss4

    def test(self, file_name):
        """
        1エポック分のテストを行う
        :param file_name: 出力ファイル名
        :return: なし
        """
        # 1つずつテストデータを取り出し、テストを行う
        for i in six.moves.range(self.num_test_data()):
            # テストを行うデータ
            i_input, i_label = self.convert(self.test_sentences[i], self.test_labels[i])
            i_input = np.asarray(i_input).astype(np.float32)
            i_input = chainer.Variable(self.xp.asarray(i_input))

            # ラベルの予測
            with chainer.using_config('train', False):
                value_until_l2 = self.model.fwd_until_l2(i_input)
                pred_labels1 = self.model.fwd1(value_until_l2)
                pred_labels2 = self.model.fwd2(value_until_l2)
                pred_labels3 = self.model.fwd3(value_until_l2)
                pred_labels4 = self.model.fwd4(value_until_l2)

            # 出力
            self.output(file_name=file_name,
                        sentence=self.test_sentences[i],
                        corr_label=self.test_labels[i],
                        pred_labels1=pred_labels1,
                        pred_labels2=pred_labels2,
                        pred_labels3=pred_labels3,
                        pred_labels4=pred_labels4)

    def convert(self, sentence, label):
        """
        文章からベクトルを生成する
        :param sentence: 文章
        :param label: ラベル
        :return: (入力ベクトルリスト, ラベルリスト)
        """
        # 入力ベクトルリストを求める
        inputs = vectorizer.sentence_vector(sentence, self.window_size)

        # vectorsと同じ要素数のラベルリストを生成
        labels = [[label] for _ in range(len(inputs))]

        return inputs, labels

    def output(self, file_name, sentence, corr_label, pred_labels1, pred_labels2, pred_labels3, pred_labels4):
        """
        テストを行なった結果をファイルに書き出す
        :param file_name: 出力ファイル名
        :param sentence: 文章
        :param corr_label: 正解ラベル
        :param pred_labels1: ラベルが1以上の予測確率
        :param pred_labels2: ラベルが2以上の予測確率
        :param pred_labels3: ラベルが3以上の予測確率
        :param pred_labels4: ラベルが4以上の予測確率
        :return: なし
        """
        # フレーズリストを出す
        phrases = spliter.phrases(sentence, self.window_size)

        # 出力
        with open(file_name, 'a') as f:
            f.write(str(corr_label) + '\t' + sentence + '\n')

            # 各フレーズに関する出力をまとめる
            for i in range(len(phrases)):
                pred_label1 = [str(label_pred) for label_pred in pred_labels1.data[i]]
                pred_label2 = [str(label_pred) for label_pred in pred_labels2.data[i]]
                pred_label3 = [str(label_pred) for label_pred in pred_labels3.data[i]]
                pred_label4 = [str(label_pred) for label_pred in pred_labels4.data[i]]
                f.write('\t'.join(pred_label1) + '\t' +
                        '\t'.join(pred_label2) + '\t' +
                        '\t'.join(pred_label3) + '\t' +
                        '\t'.join(pred_label4) + '\t' +
                        ' '.join(phrases[i]) + '\n')

        # 各文章の終わりに空行を入れる
        with open(file_name, 'a') as f:
            f.write('\n')

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
        :param file_name: モデルファイル名
        :return: なし
        """
        # モデルのロード
        serializers.load_npz(file_name, self.model)

        # GPU設定
        if self.gpu >= 0:
            chainer.cuda.get_device_from_id(self.gpu).use()
            cuda.check_cuda_available()
            self.model.to_gpu()

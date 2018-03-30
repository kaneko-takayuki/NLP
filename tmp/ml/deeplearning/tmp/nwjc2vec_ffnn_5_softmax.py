# -*- coding: utf-8 -*-

import chainer
import numpy as np
import six
from chainer import cuda
from chainer import functions as F
from chainer import optimizers
from jconvertor import spliter
from jconvertor import vectorizer

from tmp.ml import DLBases
from tmp.ml import ffnn


class NWJCS2VECFFNNSOFTMAX(DLBases):
    def __init__(self, n_in, n_mid, batchsize, gpu=-1, window_size=1):
        DLBases.__init__(self, batchsize=batchsize, gpu=gpu)

        # モデル構築
        self.model = ffnn.FFNN(n_in, n_mid, 5)

        # モデルに対するGPU設定
        if gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            cuda.check_cuda_available()
            self.model.to_gpu()

        # 最適化手法をAdamに設定
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

        # パラメータ保持
        self.window_size = window_size

    def train(self):
        """
        1エポック分の学習を行う
        :return: 合計誤差
        """
        # ランダム配列作成
        perm = np.random.permutation(self.num_train_data())

        # 誤差の総和
        sum_loss = 0

        # batchsize個ずつのミニバッチを作成し、その単位で学習を行う
        for i in six.moves.range(0, self.num_train_data(), self.batchsize):
            train_inputs = []  # 学習データのミニバッチ
            train_labels = []  # 学習ラベルのミニバッチ

            # ランダムにbatchsize個のデータを取り出し、convertで設定した方法で学習データを作り出す
            for j in six.moves.range(i, i + self.batchsize):
                if j >= self.num_train_data():
                    break  # TODO: データの変換処理を、リスト単位で行えるようにする
                j_inputs, j_labels = self.convert(self.train_sentences[perm[j]], self.train_labels[perm[j]])
                train_inputs.extend(j_inputs)
                train_labels.extend(j_labels)

            # 学習データの型を学習用に変更
            train_inputs = np.asarray(train_inputs).astype(np.float32)
            variable_train_inputs = chainer.Variable(self.xp.asarray(train_inputs))
            train_labels = np.asarray(train_labels).astype(np.int32)  # 誤差関数をsoftmax_cross_entropyにするので教師データはint
            variable_train_labels = chainer.Variable(self.xp.asarray(train_labels))

            # 学習処理
            with chainer.using_config('train', True):
                # 勾配初期化
                self.model.cleargrads()

                # フォワード処理
                _, loss = self.model(variable_train_inputs, variable_train_labels)
                sum_loss += loss.data

                # 重み更新
                self.optimizer.update()

        return sum_loss

    def dev(self, patience):
        """
        検証用データで精度を計算する
        :param patience: 様子見の回数
        :return: 検証データに対する正答率, early_stopping_flag
        """
        # 確率計算用に出力結果を保持する
        sentences_probability = []

        # 1つずつテストデータを取り出し、テストを行う
        for i in six.moves.range(self.num_dev_data()):
            # テストを行うデータ
            i_input, i_label = self.convert(self.dev_sentences[i], self.dev_labels[i])
            i_input = np.asarray(i_input).astype(np.float32)
            variable_i_input = chainer.Variable(self.xp.asarray(i_input))

            # ラベルの予測
            with chainer.using_config('train', False):
                i_probability = F.softmax(self.model.fwd(variable_i_input))
                sentences_probability.append(i_probability)

        # 検証データに対して、正答率を計算する
        dev_accuracy = self.calculate_accuracy(self.dev_labels, sentences_probability)

        self.dev_accuracy.append(dev_accuracy)

        return dev_accuracy, self.early_stopping(patience)

    def test(self, file_name):
        """
        1エポック分のテストを行う
        :param file_name: 出力ファイル名
        :return: テストデータに対する正答率
        """
        # 出力ファイルを空にする
        with open(file_name, 'w'):
            pass

        # 確率計算用に出力結果を保持する
        sentences_probability = []

        # 1つずつテストデータを取り出し、テストを行う
        for i in six.moves.range(self.num_test_data()):
            # テストを行うデータ
            i_input, i_label = self.convert(self.test_sentences[i], self.test_labels[i])
            i_input = np.asarray(i_input).astype(np.float32)
            variable_i_input = chainer.Variable(self.xp.asarray(i_input))

            # ラベルの予測
            with chainer.using_config('train', False):
                phrases_probability = F.softmax(self.model.fwd(variable_i_input))
                sentences_probability.append(phrases_probability)

        # 検証データに対して、正答率を計算する
        test_accuracy = self.calculate_accuracy(self.test_labels, sentences_probability)

        return test_accuracy

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
        labels = [label for _ in range(len(inputs))]

        return inputs, labels

    def calculate_accuracy(self, correct_labels, probabilities):
        # データ数を確認
        num_data = len(correct_labels)

        num_correct = 0  # 正解した数
        num_wrong = 0  # 間違えた数

        for i in range(num_data):
            pred_label = self.consult_softmax(probabilities[i])
            if pred_label == correct_labels[i]:
                num_correct += 1
            else:
                num_wrong += 1

        return float(num_correct) / float(num_correct + num_wrong)

    def consult_softmax(self, phrases_probability):
        """
        文章から得られる出力値について、多数決によって合議を行い、予測ラベルを返す
        :param phrases_probability: ある文章の各フレーズに対する出力リスト
        :return: 予測ラベル
        """
        # フレーズの数
        num_phrase = len(phrases_probability)

        max_phrase_probability = 0.0
        pred_sentence_label = 0

        # フレーズ毎に参照していく
        for i in range(num_phrase):
            # それぞれの予測確率を見ながら、0.5を閾値として分類してフレーズの予測ラベルを求める
            for j in range(5):
                if max_phrase_probability < phrases_probability[i][j]:
                    max_phrase_probability = phrases_probability[i][j]
                    pred_sentence_label = j

        return pred_sentence_label

    def output(self, file_name, sentence, corr_label, phrases_probability):
        """
        テストを行なった結果をファイルに書き出す
        :param file_name: 出力ファイル名
        :param sentence: 文章
        :param corr_label: 正解ラベル
        :param phrases_probability: ある文章の各フレーズに対する出力リスト
        :return: なし
        """
        # フレーズリストを出す
        phrases = spliter.phrases(sentence, self.window_size)

        # 文に対する予測ラベルを出す
        pred_label = self.consult_softmax(phrases_probability)

        # 出力
        with open(file_name, 'a') as f:
            f.write(str(corr_label) + '\t' + str(pred_label) + '\t' + sentence + '\n')

            # 各フレーズに関する出力をまとめる
            for i in range(len(phrases)):
                f.write('\t'.join(phrases_probability) +
                        ' '.join(phrases[i]) + '\n')

        # 各文章の終わりに空行を入れる
        with open(file_name, 'a') as f:
            f.write('\n')

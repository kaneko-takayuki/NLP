# -*- coding: utf-8 -*-

import copy
import numpy as np
import six
import chainer
from chainer import optimizers
from chainer import cuda

from ml.deeplearning.dlbase import DLBases
from ml.deeplearning.model import sigmoid5
from econvertor import spliter
from econvertor import vectorizer


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


class GNEWS2VECSigmoid5MAJORITY(DLBases):
    def __init__(self, n_in, n_mid, batchsize, gpu=-1, window_size=1):
        DLBases.__init__(self, batchsize=batchsize, gpu=gpu)

        # モデル構築
        self.model = sigmoid5.SIGMOID5(n_in, n_mid)

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
            train_labels = np.asarray(train_labels).astype(np.float32)

            # 学習処理
            with chainer.using_config('train', True):
                # 勾配初期化
                self.model.cleargrads()

                # 第2層までは処理が共通なので、先に計算して結果を保持しておく
                h2 = self.model.fwd_until_l2(variable_train_inputs)

                # ラベルが1以上かどうかの学習
                variable_train_labels = chainer.Variable(self.xp.asarray(threshold(train_labels, 0)))
                _, loss1 = self.model.loss1(h2, variable_train_labels)
                loss1.backward()

                # ラベルが2以上かどうかの学習
                variable_train_labels = chainer.Variable(self.xp.asarray(threshold(train_labels, 1)))
                _, loss2 = self.model.loss2(h2, variable_train_labels)
                loss2.backward()

                # ラベルが3以上かどうかの学習
                variable_train_labels = chainer.Variable(self.xp.asarray(threshold(train_labels, 2)))
                _, loss3 = self.model.loss3(h2, variable_train_labels)
                loss3.backward()

                # ラベルが4以上かどうかの学習
                variable_train_labels = chainer.Variable(self.xp.asarray(threshold(train_labels, 3)))
                _, loss4 = self.model.loss4(h2, variable_train_labels)
                loss4.backward()

                # 重み更新
                self.optimizer.update()

                # sum_lossは全ての出力の誤差総和とする
                sum_loss += (loss1.data + loss2.data + loss3.data + loss4.data)

        return sum_loss

    def dev(self, patience):
        """
        検証用データで精度を計算する
        :param patience: 様子見の回数
        :return: 検証データに対する正答率, early_stopping_flag
        """
        # 確率計算用に出力結果を保持する
        pred_labels1 = []
        pred_labels2 = []
        pred_labels3 = []
        pred_labels4 = []

        # 1つずつテストデータを取り出し、テストを行う
        for i in six.moves.range(self.num_dev_data()):
            # テストを行うデータ
            i_input, i_label = self.convert(self.dev_sentences[i], self.dev_labels[i])
            i_input = np.asarray(i_input).astype(np.float32)
            variable_i_input = chainer.Variable(self.xp.asarray(i_input))

            # ラベルの予測
            with chainer.using_config('train', False):
                # 第2層までは処理が共通なので、先に計算して結果を保持しておく
                h2 = self.model.fwd_until_l2(variable_i_input)

                pred_labels1.append(self.model.fwd1(h2).data)
                pred_labels2.append(self.model.fwd2(h2).data)
                pred_labels3.append(self.model.fwd3(h2).data)
                pred_labels4.append(self.model.fwd4(h2).data)

        # 検証データに対して、正答率を計算する
        dev_accuracy = self.calculate_accuracy(self.dev_labels,
                                               pred_labels1,
                                               pred_labels2,
                                               pred_labels3,
                                               pred_labels4)
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
        pred_labels1 = []
        pred_labels2 = []
        pred_labels3 = []
        pred_labels4 = []

        # 1つずつテストデータを取り出し、テストを行う
        for i in six.moves.range(self.num_test_data()):
            # テストを行うデータ
            i_input, i_label = self.convert(self.test_sentences[i], self.test_labels[i])
            i_input = np.asarray(i_input).astype(np.float32)
            variable_i_input = chainer.Variable(self.xp.asarray(i_input))

            # ラベルの予測
            with chainer.using_config('train', False):
                # 第2層までは処理が共通なので、先に計算して結果を保持しておく
                h2 = self.model.fwd_until_l2(variable_i_input)

                i_pred_labels1 = self.model.fwd1(h2).data
                i_pred_labels2 = self.model.fwd2(h2).data
                i_pred_labels3 = self.model.fwd3(h2).data
                i_pred_labels4 = self.model.fwd4(h2).data

                # 正答率計算用に出力値を保持しておく
                pred_labels1.append(i_pred_labels1)
                pred_labels2.append(i_pred_labels2)
                pred_labels3.append(i_pred_labels3)
                pred_labels4.append(i_pred_labels4)

            # 出力処理
            self.output(file_name,
                        self.test_sentences[i],
                        self.test_labels[i],
                        i_pred_labels1,
                        i_pred_labels2,
                        i_pred_labels3,
                        i_pred_labels4)

        # テストデータに対して、正答率を計算する
        test_accuracy = self.calculate_accuracy(self.test_labels,
                                                pred_labels1,
                                                pred_labels2,
                                                pred_labels3,
                                                pred_labels4)

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
        labels = [[label] for _ in range(len(inputs))]

        return inputs, labels

    def calculate_accuracy(self, correct_labels, pred_labels1, pred_labels2, pred_labels3, pred_labels4):
        # データ数を確認
        num_data = len(correct_labels)

        num_correct = 0  # 正解した数
        num_wrong = 0  # 間違えた数

        for i in range(num_data):
            pred_label = self.consult_majority(pred_labels1[i], pred_labels2[i], pred_labels3[i], pred_labels4[i])
            if pred_label == correct_labels[i]:
                num_correct += 1
            else:
                num_wrong += 1

        return float(num_correct) / float(num_correct + num_wrong)

    def consult_majority(self, pred_labels1, pred_labels2, pred_labels3, pred_labels4):
        """
        文章から得られる出力値について、多数決によって合議を行い、予測ラベルを返す
        :param pred_labels1: ラベルが1以上である予測確率リスト
        :param pred_labels2: ラベルが2以上である予測確率リスト
        :param pred_labels3: ラベルが3以上である予測確率リスト
        :param pred_labels4: ラベルが4以上である予測確率リスト
        :return: 予測ラベル
        """
        # フレーズの数
        num_phrase = len(pred_labels1)

        # 1〜5の予測ラベルの個数
        label_n = [0 for _ in range(5)]

        # フレーズ毎に参照していく
        for i in range(num_phrase):
            # それぞれの予測確率を見ながら、0.5を閾値として分類してフレーズの予測ラベルを求める
            if pred_labels1[i] < 0.5:
                phrase_label = 0
            elif pred_labels2[i] < 0.5:
                phrase_label = 1
            elif pred_labels3[i] < 0.5:
                phrase_label = 2
            elif pred_labels4[i] < 0.5:
                phrase_label = 3
            else:
                phrase_label = 4
            label_n[phrase_label] += 1

        # 1〜5の予測ラベルの個数を比較し、文章の予測ラベルを返す
        max_sentence_label_n = 0
        pred_sentence_label = 0
        for i in range(5):
            if max_sentence_label_n < label_n[i]:
                max_sentence_label_n = label_n[i]
                pred_sentence_label = i

        return pred_sentence_label

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

        # 文に対する予測ラベルを出す
        pred_label = self.consult_majority(pred_labels1, pred_labels2, pred_labels3, pred_labels4)

        # 出力
        with open(file_name, 'a') as f:
            f.write(str(corr_label) + '\t' + str(pred_label) + '\t' + sentence + '\n')

            # 各フレーズに関する出力をまとめる
            for i in range(len(phrases)):
                pred_label1 = [str(label_pred) for label_pred in pred_labels1[i]]
                pred_label2 = [str(label_pred) for label_pred in pred_labels2[i]]
                pred_label3 = [str(label_pred) for label_pred in pred_labels3[i]]
                pred_label4 = [str(label_pred) for label_pred in pred_labels4[i]]
                f.write('\t'.join(pred_label1) + '\t' +
                        '\t'.join(pred_label2) + '\t' +
                        '\t'.join(pred_label3) + '\t' +
                        '\t'.join(pred_label4) + '\t' +
                        ' '.join(phrases[i]) + '\n')

        # 各文章の終わりに空行を入れる
        with open(file_name, 'a') as f:
            f.write('\n')

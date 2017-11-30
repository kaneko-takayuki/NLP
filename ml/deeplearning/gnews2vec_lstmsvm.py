# -*- coding: utf-8 -*-

import numpy as np
import six
import chainer
from chainer import optimizers
from chainer import cuda
from sklearn.svm import SVC

from ml.deeplearning.dlbase import DLBases
from ml.deeplearning.model import lstm
from econvertor import spliter
from econvertor import vectorizer


class GNEWS2VECLSTMSVM(DLBases):
    def __init__(self, n_in, n_mid, n_out, batchsize, gpu=-1, kernel='linear'):
        DLBases.__init__(self, batchsize=batchsize, gpu=gpu)

        # モデル構築
        self.model = lstm.LSTM(n_in, n_mid, n_out)

        # モデルに対するGPU設定
        if gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            cuda.check_cuda_available()
            self.model.to_gpu()

        # 最適化手法をAdamに設定
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

        # パラメータを保持
        self.kernel = kernel

    def train(self):
        """
        1エポック分の学習を行う
        :return: 合計誤差
        """
        # ランダム配列作成
        perm = np.random.permutation(self.num_train_data())

        # 誤差の総和
        sum_loss = 0

        # ランダムに順番に処理していく
        for i in six.moves.range(self.num_train_data()):
            # 勾配と記憶変数を初期化
            self.model.cleargrads()
            self.model.reset_state()

            # perm[i]番目のデータについて、学習ベクトルを作り出す
            i_inputs, i_labels = self.convert(self.train_sentences[perm[i]], self.train_labels[perm[i]])

            # 学習用の型に変換する
            variable_i_inputs = chainer.Variable(self.xp.asarray(np.asarray(i_inputs).astype(np.float32)))
            variable_i_labels = chainer.Variable(self.xp.asarray(np.asarray(i_labels).astype(np.int32)))

            # 学習処理
            with chainer.using_config('train', True):
                loss = self.model(variable_i_inputs, variable_i_labels)
                sum_loss += loss.data

            # 重み更新
            loss.backward()
            self.optimizer.update()

        return sum_loss

    def dev(self, patience):
        """
        検証用データで精度を計算する
        :param patience: 様子見の回数
        :return: 検証データに対する正答率, early_stopping_flag
        """
        # SVMモデル
        svm_model = SVC(kernel=self.kernel)

        # 学習データを用意
        train_inputs = []
        train_labels = []
        for i in six.moves.range(self.num_train_data()):
            i_input, _ = self.convert(self.train_sentences[i], self.train_labels[i])
            variable_i_input = chainer.Variable(self.xp.asarray(np.asarray(i_input).astype(np.float32)))
            sentence_vector = np.asarray(self.model.get_compression_vector(variable_i_input).data)
            train_inputs.append(sentence_vector)
            train_labels.append(self.train_labels[i])

        # 学習
        svm_model.fit(train_inputs, train_labels)
        del train_inputs  # メモリ意識
        del train_labels

        # 検証データに対して、正答率を計算する
        dev_accuracy = self.calculate_accuracy(svm_model, self.dev_sentences, self.dev_labels)
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
            i_input = chainer.Variable(self.xp.asarray(i_input))

            # ラベルの予測
            with chainer.using_config('train', False):
                value_until_l2 = self.model.fwd_until_l2(i_input)
                i_pred_labels1 = self.model.fwd1(value_until_l2).data
                i_pred_labels2 = self.model.fwd2(value_until_l2).data
                i_pred_labels3 = self.model.fwd3(value_until_l2).data
                i_pred_labels4 = self.model.fwd4(value_until_l2).data

                # 正答率計算用に出力値を保持しておく
                pred_labels1.append(i_pred_labels1)
                pred_labels2.append(i_pred_labels2)
                pred_labels3.append(i_pred_labels3)
                pred_labels4.append(i_pred_labels4)

            # 出力処理
            self.output(file_name, self.test_sentences[i], self.test_labels[i], i_pred_labels1, i_pred_labels2, i_pred_labels3, i_pred_labels4)

        # 検証データに対して、正答率を計算する
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
        # 最後だけlabelで、それ以外は-1を入れる(-1の箇所は、無視されるから)
        labels = [-1 for _ in range(len(inputs))]
        labels.append(label)

        return inputs, labels

    def calculate_accuracy(self, svm_model, test_sentences, test_labels):
        # テストデータ数
        num_test = len(test_sentences)

        # 学習データを用意
        train_inputs = []
        train_labels = []
        for i in six.moves.range(num_test):
            i_input, _ = self.convert(test_sentences[i], test_labels)
            variable_i_input = chainer.Variable(self.xp.asarray(np.asarray(i_input).astype(np.float32)))
            sentence_vector = np.asarray(self.model.get_compression_vector(variable_i_input).data)
            train_inputs.append(sentence_vector)
            train_labels.append(test_labels[i])

        # テスト
        test_result = svm_model.predict(train_inputs)

        # TODO: テスト結果を確認する

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

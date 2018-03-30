# -*- coding: utf-8 -*-

import numpy as np
from econvertor import spliter
from econvertor import vectorizer
from sklearn import svm

from tmp.ml import MLBases


class EBOWSVM(MLBases):
    def __init__(self, window_size):
        MLBases.__init__(self)
        self.window_size = window_size
        self.model = svm.SVC()

    def train(self):
        """
        SVMによる学習を行う
        :return: 
        """
        # 実際に学習させるデータとラベル
        train_inputs = []
        train_labels = []

        # batchsize個毎データを取り出し学習を行う
        for (sentence, label) in zip(self.train_sentences, self.train_labels):
            i_input, i_label = self.convert(sentence, label)
            train_inputs.extend(i_input)
            train_labels.extend(i_label)

        # リストからnumpy化
        train_inputs = np.asarray(train_inputs).astype(np.float32)
        train_labels = np.asarray(train_labels).astype(np.int32)

        # 学習
        self.model.fit(train_inputs, train_labels)

    def test(self, file_name):
        """
        SVMによるテストを行う
        :param file_name: 出力ファイル名
        :return: なし
        """
        # 1つずつデータを取り出しテストを行う
        for (sentence, label) in zip(self.test_sentences, self.test_labels):
            # テストを行うデータ
            i_input, i_label = self.convert(sentence, label)
            i_input = np.asarray(i_input).astype(np.float32)

            # ラベルの予測
            pred_label = self.model.predict(i_input)

            # 出力
            self.output(file_name, sentence, label, pred_label)

    def convert(self, sentence, label):
        """
        文章からベクトルを生成する
        :param sentence: 英語文章
        :param label: ラベル
        :return: (入力ベクトルリスト, ラベルリスト)
        """
        # 入力ベクトルリストを取得
        inputs = vectorizer.sentence_vector(sentence, self.window_size)

        # vectorsと同じ要素数のラベルリストを生成
        labels = [label for _ in range(len(inputs))]

        return inputs, labels

    def output(self, file_name, sentence, corr_label, pred_labels):
        """
        結果をファイルに出力する
        :param file_name: 出力ファイル
        :param sentence: 文章
        :param corr_label: 正解ラベル
        :param pred_labels: 予測確率
        :return: なし
        """
        # フレーズリストを取得する
        phrases = spliter.phrases(sentence, self.window_size)

        # 出力
        with open(file_name, 'a') as f:
            f.write(str(corr_label) + '\t' + sentence + '\n')

            # 各フレーズに関する出力をまとめる
            for (phrase, pred_label) in zip(phrases, pred_labels):
                f.write(str(pred_label) + '\t' + ' '.join(phrase) + '\n')

        # 各文章の終わりに空行を入れる
        with open(file_name, 'a') as f:
            f.write('\n')
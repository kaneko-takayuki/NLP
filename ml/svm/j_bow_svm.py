# -*- coding: utf-8 -*-

import numpy as np
from sklearn import svm

from ml.base import MLBases
from jconvertor.bow import func


class JBOWSVM(MLBases):
    def __init__(self):
        MLBases.__init__(self)
        self.model = svm.SVC()

    def train(self):
        """
        SVMによる学習を行う
        :return: 
        """
        # 実際に学習させるデータとラベル
        train_inputs = []
        train_labels = []

        # batchsize個毎にデータを取り出し、学習を行う
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

        # 1つずつデータを取り出し、テストを行う
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
        :param sentence: 日本語文章
        :param label: ラベル
        :return: (入力ベクトルリスト, ラベルリスト)
        """
        # Bag of Words単語リストを作成
        for _sentence in self.train_sentences:
            func.add_dir(_sentence)
        for _sentence in self.test_sentences:
            func.add_dir(_sentence)
        # 入力ベクトルリストを取得
        inputs = [func.bow(sentence)]

        # vectorsと同じ要素数のラベルリストを生成
        labels = [label]

        return inputs, labels

    def output(self, file_name, sentence, corr_label, pred_labels):
        """
        結果をファイルに出力する
        :param file_name: 出力ファイル名
        :param sentence: 文章
        :param corr_label: 正解ラベル
        :param pred_labels: 予測確率
        :return: なし
        """
        # 素性がBoWなので、予測ラベルは1文につき1つだけ
        with open(file_name, 'a') as f:
            f.write(str(corr_label) + '\t' + str(pred_labels[0]) + '\t' + sentence + '\n')

        # 各文章の終わりに空行を入れる
        with open(file_name, 'a') as f:
            f.write('\n')

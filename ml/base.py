# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod


class MLBases(metaclass=ABCMeta):
    def __init__(self):
        # 学習データ
        self.train_sentences = []
        self.train_labels = []

        # テストデータ
        self.test_sentences = []
        self.test_labels = []

    def set_train_data(self, ts, tl):
        """
        学習データをセット
        :param ts: 学習する文章リスト <class: 'list'>
        :param tl: 学習するラベルリスト <class: 'list'>
        :return: なし
        """
        self.train_sentences = ts
        self.train_labels = tl

    def set_test_data(self, ts, tl):
        """
        テストデータをセット
        :param ts: テストする文章リスト <class: 'list'>
        :param tl: テストするラベルリスト <class: 'list'>
        :return: なし
        """
        self.test_sentences = ts
        self.test_labels = tl

    def num_train_data(self):
        """
        学習データ数を返す
        :return: 学習データ数
        """
        return len(self.train_sentences)

    def num_test_data(self):
        """
        テストデータ数を返す
        :return: テストデータ数
        """
        return len(self.test_sentences)

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self, file_name):
        pass

    @abstractmethod
    def convert(self, sentence, label):
        pass

    @abstractmethod
    def output(self, file_name, sentence, corr_label, pred_labels):
        pass

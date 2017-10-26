# -*- coding: utf-8 -*-

from abc import ABCMeta
import smtplib
from email.mime.text import MIMEText

import BagofWords
import EnglishSentenceConvertor


class RunnerBases(metaclass=ABCMeta):

    def __init__(self):
        # 学習データ
        self.train_sentences = []
        self.train_labels = []
        self.train_N = 0

        # テストデータ
        self.test_sentences = []
        self.test_labels = []
        self.test_N = 0

        # 学習・テストデータ
        self.all_sentences = []
        self.all_labels = []
        self.all_N = 0

        # Bag of Words
        self.bow = BagofWords.BagofWords()

    def set_train_data(self, data):
        """
        学習データをセットする
        :param data: [[文1, ラベル1], [文2, ラベル2], [文3, ラベル3]...[文N, ラベルN]]
        :return: なし
        """
        # 学習データをセットする
        # この場で文を分散表現に展開すると、容量が膨大になる可能性があるから、
        # 速度は低下するが学習時・テスト時の直前に必要な分だけ分散表現に展開する
        self.train_sentences = []
        self.train_labels = []
        self.train_N = len(data)
        for one_data in data:
            self.train_sentences.append(EnglishSentenceConvertor.EnglishSentenceConvertor(one_data[0]))
            self.train_labels.append(one_data[1])

        # Bag of Wordsに文データを追加
        for sentence in self.train_sentences:
            self.bow.add_dir(sentence=sentence())

    def set_test_data(self, data):
        """
        テストデータをセットする
        :param data: [[文1, ラベル1], [文2, ラベル2], [文3, ラベル3]...[文N, ラベルN]]
        :return: なし
        """
        # テストデータをセットする
        self.test_N = len(data)
        self.test_sentences = []
        self.test_labels = []
        for one_data in data:
            self.test_sentences.append(EnglishSentenceConvertor.EnglishSentenceConvertor(one_data[0]))
            self.test_labels.append(one_data[1])

        # Bag of Words用に文データを追加
        for sentence in self.test_sentences:
            self.bow.add_dir(sentence=sentence())

    def set_all_data(self, data):
        """
        5分割交差検定などで学習用とテスト用で、ランダムにデータを振り分けたい時用
        :param data: [[文1, ラベル1], [文2, ラベル2], [文3, ラベル3]...[文N, ラベルN]]
        :return: なし
        """
        for one_data in data:
            self.all_sentences.append(one_data[0])
            self.all_labels.append(one_data[1])
        self.all_N = len(self.all_sentences)

        # Bag of Words用に文データを追加
        for sentence in self.all_sentences:
            self.bow.add_dir(sentence=sentence)

    def send_mail(self, subject, body):
        """
        メールを送信する
        :param subject: 題名
        :param body: 本文
        :return: なし
        """
        host, port = 'smtp.gmail.com', 465
        username, password = 'g.knk.suteaka@gmail.com', 'suteakadesu'
        to = 'g.knk.9410@gmail.com'

        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = username
        msg['To'] = to

        smtp = smtplib.SMTP_SSL(host, port)
        smtp.ehlo()
        smtp.login(username, password)
        smtp.mail(username)
        smtp.rcpt(to)
        smtp.data(msg.as_string())
        smtp.quit()

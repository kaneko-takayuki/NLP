# -*- coding: utf-8 -*-

from sklearn import svm

from tmp.ml import MLBases


class LSTMSVM(MLBases):
    def __init__(self, kernel):
        MLBases.__init__(self)

        self.train_vectors = []
        self.test_vectors = []
        self.model = svm.SVC(kernel=kernel)

    def set_train_data(self, ts, tv, tl):
        """
        学習データをセット
        :param ts: 学習する文章リスト <class: 'list'>
        :param tv: 学習するベクトルリスト <class: 'list'>
        :param tl: 学習するラベルリスト <class: 'list'>
        :return: なし
        """
        self.train_sentences = ts
        self.train_vectors = tv
        self.train_labels = tl

    def set_test_data(self, ts, tv, tl):
        """
        テストデータをセット
        :param ts: 学習する文章リスト <class: 'list'>
        :param tv: テストするベクトル <class: 'list'>
        :param tl: テストするラベルリスト <class: 'list'>
        :return: なし
        """
        self.test_sentences = ts
        self.test_vectors = tv
        self.test_labels = tl

    def train(self):
        """
        1エポック分の学習を行う
        :return: 
        """
        self.model.fit(self.train_vectors, self.train_labels)

    def test(self, file_name):
        """
        1エポック分のテストを行う
        :params file_name: 出力ファイル名
        :return: なし
        """
        results = self.model.predict(self.test_vectors)
        self.output(file_name, self.test_sentences, self.test_labels, results)

    def output(self, file_name, sentences, labels, results):
        with open(file_name, 'w') as f:
            for i in range(len(sentences)):
                sentence = sentences[i]
                label = labels[i]
                result = results[i]
                f.write(str(label) + '\t' + str(result) + '\t' + str(sentence) + '\n')

    def convert(self):
        pass

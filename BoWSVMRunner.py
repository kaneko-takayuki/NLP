# -*- coding: utf-8 -*-

import numpy as np
from sklearn import svm

import RunnerBases


class BoWSVMRunner(RunnerBases.RunnerBases):
    """
    Bag of Wordsを素性として、SVMによる学習・テストを行うクラス
    """
    def __init__(self):
        RunnerBases.RunnerBases.__init__(self)
        self.model = None

    def train(self):
        """
        SVMによる学習を行う
        :return: なし
        """
        # SVMモデル
        self.model = svm.SVC(kernel='linear')

        # 学習ベクトル、ラベルをまとめるリスト
        _train_vectors = []
        _train_labels = []

        # 学習するベクトルとラベルの組を用意
        for (sentence, label) in zip(self.train_sentences, self.train_labels):
            vec = self.bow.get_bow(sentence=sentence())
            _train_vectors.append(vec)
            _train_labels.append(label)

        # numpyリストに変換
        _train_vectors = np.asarray(_train_vectors)
        _train_labels = np.asarray(_train_labels)

        # 学習
        self.model.fit(_train_vectors, _train_labels)

    def test(self, file_name=""):
        """
        SVMによるテストを行う
        :param file_name: 出力ファイル
        :return: なし
        """
        # 学習済みか確認
        if not self.model:
            print("学習がまだ行われていません")
            return

        # 出力ファイルの初期化
        if file_name != "":
            with open(file_name, "w"):
                pass

        # correct_n: 正解数
        # wrong_n: 不正解数
        correct_n = 0
        wrong_n = 0

        for (sentence, label) in zip(self.test_sentences, self.test_labels):
            # 文からBag of Wordsを取得
            vec = self.bow.get_bow(sentence=sentence())
            # 予測
            predict = self.model.predict([vec])

            # 正解かどうか検証
            if predict[0] == label:
                correct_n += 1
            else:
                wrong_n += 1

            # ファイルに出力
            if file_name != "":
                with open(file_name, "a") as f:
                    f.write(str(label) + " " + str(predict[0]) + " ")
                    f.write(sentence())

        # 正解率を出力
        print("{} / {}".format(correct_n, (correct_n + wrong_n)))
        print(correct_n/(correct_n + wrong_n))

    def cross_validation(self, k=5, dir_name=""):
        """
        k分割交差検定を行う
        :param k: k分割
        :param dir_name: 出力ディレクトリ名
        :return: なし
        """

        # 文・ラベルをランダムに以下のようにk分割する
        # unprocessed_data: [[[文1_1, ラベル1_1], ..., [文1_n, ラベル1_n]], [[文2_1, ラベル2_1], ..., [文2_n, ラベル2_n]], ..., [文・ラベルセットk]]
        perm = np.random.permutation(self.all_N)
        data = []
        for i in perm:
            data.append([self.all_sentences[i], self.all_labels[i]])
        split_num = int(self.all_N / k)
        data = [data[x:x + split_num] for x in range(0, self.all_N, split_num)]

        # k分割交差検定
        for i in range(k):
            print("{}/{}回目交差検定".format(i + 1, k))
            # 学習
            _train_data = []
            _test_data = []

            # 学習データ・テストデータを分ける
            for j in range(k):
                if i != j:
                    _train_data.extend(data[j])
                else:
                    _test_data.extend(data[j])

            # 学習データ・テストデータをセット
            self.set_train_data(_train_data)
            self.set_test_data(_test_data)

            # 学習
            self.train()

            # テスト
            output_file = "" if (dir_name == "") else (dir_name+"cross-validation"+str(i)+".txt")
            self.test(file_name=output_file)

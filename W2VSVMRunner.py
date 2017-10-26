# -*- coding: utf-8 -*-

import numpy as np
from sklearn import svm

import RunnerBases


class W2VSVMRunner(RunnerBases.RunnerBases):
    """
    Word2Vecを素性として、SVMによる学習・テストを行うクラス
    """
    def __init__(self):
        RunnerBases.RunnerBases.__init__(self)
        self.model = None

    def train(self, w=0):
        """
        SVMによる学習を行う
        :param w: 区切るウィンドウサイズ
        :return: なし
        """
        # SVMモデル
        self.model = svm.SVC()

        # 学習ベクトル、ラベルをまとめるリスト
        _train_vectors = []
        _train_labels = []

        # 学習するベクトルとラベルの組を用意
        for (sentence, label) in zip(self.train_sentences, self.train_labels):
            vectors = sentence.get_vectors(window_size=w)
            _train_vectors.extend(vectors)
            _train_labels.extend([label for _ in range(len(vectors))])

        # numpyリストに変換
        _train_vectors = np.asarray(_train_vectors)
        _train_labels = np.asarray(_train_labels)

        # 学習
        self.model.fit(_train_vectors, _train_labels)

    def test(self, w=0, file_name=""):
        """
        SVMによるテストを行う
        :param w: ウィンドウサイズ
        :param file_name: 出力ディレクトリ
        :return: なし
        """
        # 学習済みか確認
        if not self.model:
            print("学習がまだ行われていません")
            return

        # 出力ファイルを新規作成
        open(file_name, "w")

        for (i, sentence) in enumerate(self.test_sentences):
            # 文章を出力
            with open(file_name, "a") as o:
                o.write(str(self.test_labels[i]) + "\t" + self.test_sentences[i]())

            # テストデータを準備してテスト
            input_vector = sentence.get_vectors(window_size=w)
            # 分かち書きリスト
            window_wakati = self.test_sentences[i].get_wakati(window_size=w)

            for (wakati, vec) in zip(window_wakati, input_vector):
                # 予測ラベル・超平面までの距離を計算
                predict = self.model.predict([vec])[0]
                distance = self.model.decision_function([vec])[0]

                with open(file_name, "a") as o:
                    o.write(str(predict) + "\t" + str(distance) + "\t" + wakati + "\n")

            with open(file_name, "a") as o:
                o.write("\n")

    def cross_validation(self, w=0, k=5, dir_name=""):
        """
        k分割交差検定を行う
        :param w: ウィンドウサイズ
        :param k: k分割
        :param dir_name: 出力ディレクトリ名
        :return: なし
        """
        if w == 0:
            print("ウィンドウサイズを指定してください")
            return

        # 文・ラベルをランダムに以下のようにk分割する
        perm = np.random.permutation(self.all_N)
        data = []
        for i in perm:
            data.append([self.all_sentences[i], self.all_labels[i]])
        split_num = int(self.all_N / k)
        data = [data[x:x + split_num] for x in range(0, self.all_N, split_num)]

        # k分割交差検定
        for i in range(k):
            print("{}/{}回目交差検定".format(i+1, k))
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
            self.train(w=w)

            # テスト
            output_file = "" if (dir_name == "") else (dir_name+"cross-validation"+str(i)+".txt")
            self.test(w=w, file_name=output_file)

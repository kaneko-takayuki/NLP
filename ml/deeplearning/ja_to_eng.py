# -*- coding: utf-8 -*-

import numpy as np
from numpy import dot
import six
import random
import chainer
from chainer import serializers
from ml.deeplearning.model import ffnn2
from chainer import optimizers
from chainer import cuda
from gensim import matutils

from jconvertor import vectorizer as jvec
from econvertor import vectorizer as evec
from econvertor import word2vec as ew2v


class JAtoENG:
    def __init__(self, n_in, n_mid, n_out, batchsize, gpu=-1):
        # 学習データ
        self.train_words_dir = {}

        # モデル構築
        self.model = ffnn2.FFNN(n_in, n_mid, n_out)

        # GPU設定
        if gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            cuda.check_cuda_available()
            self.model.to_gpu()

        # 最適化手法をAdamに設定
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

        # DeepLearningの設定関連
        self.xp = np if gpu < 0 else cuda.cupy
        self.batchsize = batchsize
        self.gpu = gpu

    def set_train_data(self, words_dir):
        """
        学習データをセット
        :param words_dir: 日本単語と英単語を対応させている辞書
        :return: なし
        """
        self.train_words_dir = words_dir

    def num_train_data(self):
        """
        学習データ数を返す
        :return: 学習データ数
        """
        return len(self.train_words_dir)

    def train(self):
        """
        1エポック分の学習を行う
        :return: なし
        """
        sum_loss = 0.0

        # ランダム配列作成
        perm = np.random.permutation(self.num_train_data())

        # キーリスト
        keys = list(self.train_words_dir.keys())

        # batchsize個ずつデータを入れて学習させていく
        for i in six.moves.range(0, self.num_train_data() - self.batchsize, self.batchsize):
            # 実際に学習させるデータとラベル
            train_inputs = []
            train_outputs = []

            # ランダムにbatchsize個のデータを取り出し、convertで設定した方法で学習データを作り出す
            for j in six.moves.range(i, i + self.batchsize):
                j_input, j_output = self.convert(keys[perm[j]], self.train_words_dir[keys[perm[j]]])
                train_inputs.extend(j_input)
                train_outputs.extend(j_output)

            # もし、学習するデータがない場合、学習を行わずに次のミニバッチ処理へ
            if len(train_inputs) == 0:
                continue

            # リストからnumpy化
            train_inputs = np.asarray(train_inputs).astype(np.float32)
            train_outputs = np.asarray(train_outputs).astype(np.float32)

            # 学習処理
            train_inputs = chainer.Variable(self.xp.asarray(train_inputs))
            train_outputs = chainer.Variable(self.xp.asarray(train_outputs))
            with chainer.using_config('train', True):
                self.model.cleargrads()
                try:
                    loss = self.model(train_inputs, train_outputs)
                except:
                    pass
                sum_loss += loss.data
                loss.backward()
                self.optimizer.update()

        print(sum_loss)


        # train_inputs = []
        # train_outputs = []
        #
        # j_input, j_output = self.convert('日本', 'japan')
        #
        # train_inputs.extend(j_input)
        # train_outputs.extend(j_output)
        #
        # train_inputs = np.asarray(train_inputs).astype(np.float32)
        # train_outputs = np.asarray(train_outputs).astype(np.float32)
        #
        # train_inputs = chainer.Variable(self.xp.asarray(train_inputs))
        # train_outputs = chainer.Variable(self.xp.asarray(train_outputs))
        #
        # with chainer.using_config('train', True):
        #     self.model.cleargrads()
        #     loss = self.model(train_inputs, train_outputs)
        #     loss.backward()
        #     self.optimizer.update()

    def convert(self, j_word, e_words):
        """
        日本単語と英単語を対応させたベクトルデータを返す
        (どちらかがKeyErrorになる対応データは無視)
        :param j_word: 日単語
        :param e_words: 対応する英単語リスト
        :return: 対応ベクトル
        """
        j_input = []
        j_output = []

        # 日単語について見る
        # キーがなければ空リストを返す
        j_word_vec = jvec.word_vector(j_word)
        if j_word_vec is None:
            return j_input, j_output

        # 対応する英単語一つずつ見ていき、ベクトルリストを格納
        for e_word in e_words:
            e_word_vec = evec.word_vector(e_word)
            if e_word_vec is None:
                continue
            j_input.append(j_word_vec)
            j_output.append(e_word_vec)

        return j_input, j_output

    def save(self, file_name):
        """
        モデルを保存する
        :param file_name: セーブファイル名
        :return: なし
        """
        # CPUモードで保存
        self.model.to_cpu()
        serializers.save_npz(file_name, self.model)

        # GPU設定
        if self.gpu >= 0:
            chainer.cuda.get_device_from_id(self.gpu).use()
            cuda.check_cuda_available()
            self.model.to_gpu()

    def load(self, file_name):
        """
        モデルを読み込む
        :param file_name: ロードファイル名
        :return: なし
        """
        # モデルをロードする
        serializers.load_npz(file_name, self.model)

        # GPU設定
        if self.gpu >= 0:
            chainer.cuda.get_device_from_id(self.gpu).use()
            cuda.check_cuda_available()
            self.model.to_gpu()

    def foward(self, j_word):
        """
        日本語単語を受け取り、英単語ベクトルに変換する
        :param j_word: 日単語
        :return: 英単語ベクトル
        """
        # キーが無ければ、300次元のランダムベクトルを返す
        j_word_vec = jvec.word_vector(j_word)
        if j_word_vec is None:
            return []

        # 与えられた日単語ベクトルに対する英単語ベクトルを求めて返す
        with chainer.using_config('train', False):
            j_word_vec = self.xp.asarray([j_word_vec]).astype(np.float32)
            output = self.model.fwd(j_word_vec)
            return list(output.data[0])

    def most_similar(self, j_word):
        """
        日本語単語ベクトルを英単語ベクトルに変換した後、
        どのような英単語とマッチするか求める
        :param j_word: 日単語
        :return: マッチする英単語と類似度リスト
        """
        # キーが無ければ、空リストを返す
        j_word_vec = jvec.word_vector(j_word)
        if j_word_vec is None:
            return []

        # 与えられた日単語ベクトルに対する英単語ベクトルを求める
        with chainer.using_config('train', False):
            j_word_vec = self.xp.asarray([j_word_vec]).astype(np.float32)
            output = self.model.fwd(j_word_vec)
            converted_vector = np.asarray(list(output.data[0]), np.float32)

            # 出力したベクトルに対し、マッチする英単語を求める
            match_result = ew2v.model.most_similar(positive=[converted_vector], topn=10)
            return match_result

    def similarity(self, j_word, e_word):
        """
        単語j_wordの日本語単語ベクトルを英単語ベクトルに変換した後、
        単語e_wordの単語ベクトルとのコサイン類似度を計算する
        :param j_word: 日単語
        :param e_word: 英単語
        :return: j_wordを英ベクトル化したものと、e_wordのベクトルのコサイン類似度
        """
        # キーが無ければ、0を返す
        j_word_vec = jvec.word_vector(j_word)
        e_word_vec = evec.word_vector(e_word)
        if (j_word_vec is None) or (e_word_vec is None):
            return 0.0

        # 与えられた日単語ベクトルに対する英単語ベクトルを求める
        with chainer.using_config('train', False):
            j_word_vec = self.xp.asarray([j_word_vec]).astype(np.float32)
            output = self.model.fwd(j_word_vec)
            converted_vector = np.asarray(list(output.data[0]), np.float32)

            # 出力したベクトルに対し、マッチする英単語を求める
            return dot(matutils.unitvec(converted_vector), matutils.unitvec(e_word_vec))

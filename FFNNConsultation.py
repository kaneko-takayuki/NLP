# -*- coding: utf-8 -*-

import os
import numpy as np
import six

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer import serializers
from chainer import cuda

import RunnerBases

# ライブラリのパスを通す
os.environ["PATH"] = "/usr/local/cuda-7.5/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


class FFNN(chainer.Chain):
    """
    入力ノード次元n_in、中間ノード次元n_units、出力ノード次元n_out
    の3層から成るフィードフォワードニューラルネットワーク
    """

    def __init__(self, n_in, n_units, n_out):
        super(FFNN, self).__init__(
            l1=L.Linear(n_in, n_units),
            l2=L.Linear(n_units, n_units),
            l3=L.Linear(n_units, n_out),
        )

    def __call__(self, x, y):
        return F.softmax_cross_entropy(self.fwd(x), y)

    def fwd(self, x):
        h1 = F.dropout(F.relu(self.l1(x)))
        h2 = F.dropout(F.relu(self.l2(h1)))
        return self.l3(h2)


class FFNNConsultation(RunnerBases.RunnerBases):
    """
    合議によるフィードフォワードニューラルネットワーク
    """

    def __init__(self):
        """
        コンストラクタ
        """
        RunnerBases.RunnerBases.__init__(self)

        # 変数設定
        self.model = None
        self.optimizer = None
        self.xp = None
        self.gpu = None
        self.ws = 0

    def init_model(self, n_in=1500, n_units=1000, n_out=2, window_size=5, gpu=-1):
        """
        モデルを新しく作成する
        :param n_in: 入力次元数
        :param n_units: 中間次元数
        :param n_out: 出力次元数
        :param gpu: GPUのID
        :return: なし
        """

        # モデル構築
        self.model = FFNN(n_in, n_units, n_out)

        # GPU設定
        if gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            cuda.check_cuda_available()
            self.model.to_gpu()

        self.xp = np if gpu < 0 else cuda.cupy

        # 最適化手法をAdamに設定
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

        # パラメータ保持
        self.gpu = gpu
        self.ws = window_size

    def train(self, batchsize=30):
        """
        1エポック分の学習を行う
        :return: なし
        """
        # ランダム配列
        perm = np.random.permutation(self.train_N)

        for i in six.moves.range(0, self.train_N-batchsize, batchsize):
            # 入力ベクトル・ラベルを生成する
            batch_inputs = []
            batch_labels = []

            # ランダムにbatchsize個取り出し、window-sizeで切り取り分散表現を得る
            for j in range(i, i+batchsize):
                input_vector = self.train_sentences[perm[j]].get_vectors(window_size=self.ws)
                batch_inputs.extend(input_vector)
                # 1文から取り出せた分散表現の数だけラベルを生成
                batch_labels.extend([self.train_labels[perm[j]] for _ in range(len(input_vector))])
            batch_inputs = np.asarray(batch_inputs).astype(np.float32)
            batch_labels = np.asarray(batch_labels).astype(np.int32)

            # 学習処理
            batch_inputs = chainer.Variable(self.xp.asarray(batch_inputs))
            batch_labels = chainer.Variable(self.xp.asarray(batch_labels))
            with chainer.using_config('train', True):
                self.model.cleargrads()
                loss = self.model(batch_inputs, batch_labels)
                loss.backward()
                self.optimizer.update()

    def test(self, file_name):
        """
        1エポック分のテストを行う
        :param file_name: 出力ファイル名
        :return: なし
        """
        # ファイルを新規作成
        # すでに存在している場合は上書き
        open(file_name, "w")

        for i in range(self.test_N):
            # 文章を出力
            with open(file_name, "a") as o:
                o.write(str(self.test_labels[i]) + "\t" + self.test_sentences[i]())

            # テストデータを準備してテスト
            input_vector = self.test_sentences[i].get_vectors(window_size=self.ws)
            input_vector = np.asarray(input_vector).astype(np.float32)
            inputs = chainer.Variable(self.xp.asarray(np.asarray(input_vector).astype(np.float32)))
            # 分かち書きのリスト
            window_wakati = self.test_sentences[i].get_wakati(window_size=self.ws)
            # 推測ラベルを求める
            with chainer.using_config('train', False):
                guess_labels = F.softmax(self.model.fwd(inputs))

            # それぞれ分かち書き・予測ラベルをファイルに出力
            for (wakati, guess_label) in zip(window_wakati, guess_labels.data):
                with open(file_name, "a") as o:
                    o.write("\t".join(guess_label) + "\t" + wakati + "\n")
            with open(file_name, "a") as o:
                o.write("\n")

    def save(self, file_name):
        """
        モデルのセーブを行う
        :param file_name: セーブファイル名
        :return: なし
        """
        self.model.to_cpu()  # CPUで計算できるようにしておく
        serializers.save_npz(file_name, self.model)  # npz形式で書き出し
        if self.gpu >= 0:
            chainer.cuda.get_device_from_id(self.gpu).use()
            cuda.check_cuda_available()
            self.model.to_gpu()

    def load(self, file_name):
        """
        モデルのロードを行う
        :param file_name: ロードファイル名
        :return: なし
        """

        # モデルの設定を行っている時だけ動作させる
        if self.model:
            serializers.load_npz(file_name, self.model)

        if self.gpu >= 0:
            chainer.cuda.get_device_from_id(self.gpu).use()
            cuda.check_cuda_available()
            self.model.to_gpu()

# -*- coding: utf-8 -*-

import os
import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer import cuda
from chainer import serializers

import SentenceConvertor
import RunnerBases

# ライブラリのパスを通す
os.environ["PATH"] = "/usr/local/cuda-7.5/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


class LSTM(chainer.Chain):
    """
    入力ノード次元n_in、中間ノード次元n_units、出力ノード次元n_out
    の3層から成るLSTMネットワーク
    """

    def __init__(self, n_in, n_units, n_out):
        super(LSTM, self).__init__(
            l1=L.LSTM(n_in, n_units),
            l2=L.LSTM(n_units, n_units),
            l3=L.Linear(n_units, n_out),
        )

    def __call__(self, x, y):
        return F.softmax_cross_entropy(self.fwd(x), y)

    def fwd(self, x):
        h0 = F.dropout(x)
        h1 = F.dropout(self.l1(h0))
        h2 = F.dropout(self.l2(h1))
        return self.l3(h2)

    def get_compression_vector(self, x):
        h0 = F.dropout(x)
        h1 = F.dropout(self.l1(h0))
        return self.l2(h1)

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()


class LSTMRunner(RunnerBases.RunnerBases):
    """
    LSTMニューラルネットワーク
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

    def init_model(self, n_in=1500, n_units=1000, n_out=2, gpu=-1):
        """
        モデルを新しく作成する
        :param n_in: 入力次元数
        :param n_units: 中間次元数
        :param n_out: 出力次元数
        :param gpu: GPUのID
        :return: なし
        """

        # モデル構築
        self.model = LSTM(n_in, n_units, n_out)

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

    def train(self, batchsize=30):
        """
        1エポック分の学習を行う
        :return: なし
        """

        # 1文ずつ処理していく
        for (sentence, label) in zip(self.train_sentences, self.train_labels):
            w2v_vectors = sentence.get_vectors()
            t = chainer.Variable(self.xp.asarray([label]).astype(np.int32))

            self.model.cleargrads()
            self.model.reset_state()

            for v in w2v_vectors:
                x = chainer.Variable(self.xp.asarray([v]).astype(np.float32))
                with chainer.using_config('train', True):
                    loss = self.model(x, t)

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

        for (sentence, label) in zip(self.test_sentences, self.test_labels):
            # 文中の単語ベクトルを求める
            w2v_vectors = sentence.get_vectors()
            # 内部状態を初期化
            self.model.reset_state()

            # 一単語ずつ読み込んでラベルを予想
            for v in w2v_vectors:
                x = chainer.Variable(self.xp.asarray([v]).astype(np.float32))
                with chainer.using_config('train', False):
                    predict = F.softmax(self.model.fwd(x))

            with open(file_name, "a") as o:
                o.write(str(label) + "\t" + sentence())
                # 最後の出力のみ取ってくる
                o.write(str(predict.data[-1][0]) + "\t" + str(predict.data[-1][1]) + "\n\n")

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

    def get_compression_vector(self, sentence):
        """
        文を与えて文圧縮ベクトルを返す
        :param sentence: 対象の文
        :return: 文圧縮ベクトル
        """

        # 文中の単語ベクトルを求める
        sc = SentenceConvertor.SentenceConvertor(sentence)
        w2v_vectors = sc.get_vectors()

        # 文圧縮ベクトルを求める
        for v in w2v_vectors:
            x = chainer.Variable(self.xp.asarray([v]).astype(np.float32))
            with chainer.using_config('train', False):
                compression_vector = self.model.get_compression_vector(x)

        return compression_vector.data
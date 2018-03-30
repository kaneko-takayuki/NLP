# -*- coding: utf-8 -*-

import chainer
import numpy as np
import six
from chainer import cuda
from chainer import optimizers
from econvertor import vectorizer

from tmp.ml import LSTMBases
from tmp.ml import lstm


class MakeSentenceVectorLSTM(LSTMBases):
    def __init__(self, n_in, n_mid, n_out, batchsize, gpu=-1):
        LSTMBases.__init__(self, batchsize, gpu)

        # モデル構築
        self.model = lstm.LSTM(n_in, n_mid, n_out)

        # GPU設定
        if gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            cuda.check_cuda_available()
            self.model.to_gpu()

        # 最適化手法をAdamに設定
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

    def convert(self, sentence, label):
        """
        文章からベクトルを生成する
        :param sentence: 英語文章
        :param label: ラベル
        :return: (入力ベクトルリスト, ラベルリスト)
        """
        # 入力ベクトルリストを取得
        inputs = vectorizer.sentence_vector(sentence)

        # labelを配列形式にして返す
        return inputs, [label]

    def output(self, file_name):
        """
        文を与えて文圧縮ベクトルを出力する
        :param file_name: 出力ファイル
        :return: なし
        """
        # ファイルを初期化
        with open(file_name, 'w'):
            pass

        # 1つずつデータを取り出し、テストを行う
        for i in six.moves.range(self.num_test_data()):
            # テストを行うデータ
            i_input, i_label = self.convert(self.test_sentences[i], self.test_labels[i])
            word_len = len(i_input)
            self.model.cleargrads()  # 勾配の初期化
            self.model.reset_state()  # 内部状態の初期化

            # j番目の単語について見ていく
            for j in range(word_len):
                x = chainer.Variable(self.xp.asarray(np.asarray([i_input[j]]).astype(np.float32)))

                # 最後の出力に対してのみ誤差を計算し、逆伝播
                if j == word_len - 1:
                    with chainer.using_config('train', False), open(file_name, 'a') as f:
                        compression_vector = self.model.get_compression_vector(x)
                        str_compression_vector = [str(v) for vec in compression_vector.data for v in vec]
                        f.write(str(self.test_labels[i]) + '\t' + str(self.test_sentences[i]) + '\n')
                        f.write(' '.join(str_compression_vector) + '\n')
                else:
                    with chainer.using_config('train', False):
                        self.model.fwd(x)

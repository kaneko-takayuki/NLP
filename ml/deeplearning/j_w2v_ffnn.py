# -*- coding: utf-8 -*-

import chainer
from chainer import optimizers
from chainer import cuda

from ml.deeplearning.dlbase import DLBases
from ml.deeplearning.model import ffnn
from jconvertor import spliter
from jconvertor import vectorizer


class JW2VFFNN(DLBases):
    def __init__(self, n_in, n_mid, n_out, batchsize, gpu=-1, window_size=1):
        DLBases.__init__(self, batchsize, gpu)

        # モデル構築
        self.model = ffnn.FFNN(n_in, n_mid, n_out)

        # GPU設定
        if gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            cuda.check_cuda_available()
            self.model.to_gpu()

        # 最適化手法をAdamに設定
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

        # パラメータ保持
        self.window_size = window_size

    def convert(self, sentence, label):
        """
        文章からベクトルを生成する
        :param sentence: 日本語文章
        :param label: ラベル
        :return: (入力ベクトルリスト, ラベルリスト)
        """
        # 入力ベクトルリストを取得
        inputs = vectorizer.sentence_vector(sentence, self.window_size)

        # vectorsと同じ要素数のラベルリストを生成
        labels = [label for _ in range(len(inputs))]

        return inputs, labels

    def output(self, file_name, sentence, corr_label, pred_labels):
        phrases = spliter.phrases(sentence, self.window_size)
        with open(file_name, 'a') as f:
            f.write(str(corr_label) + '\t' + sentence + '\n')

            # 各フレーズに関する出力をまとめる
            for (phrase, pred_label) in zip(phrases, pred_labels.data):
                pred_label = [str(label) for label in pred_label]
                f.write('\t'.join(pred_label) + '\t' + ' '.join(phrase) + '\n')

        # 各文章の終わりに空行を入れる
        with open(file_name, 'a') as f:
            f.write('\n')

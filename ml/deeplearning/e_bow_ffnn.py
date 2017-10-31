# -*- coding: utf-8 -*-

import chainer
from chainer import optimizers
from chainer import cuda

from ml.deeplearning.dlbase import DLBases
from ml.deeplearning.model import ffnn
from econvertor.bow import func


class EBOWFFNN(DLBases):
    def __init__(self, n_in, n_mid, n_out, batchsize, gpu=-1):
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

    def convert(self, sentence, label):
        """
        文章からベクトルを生成する
        :param sentence: 英語文章
        :param label: ラベル
        :return: (入力ベクトルリスト, ラベルリスト)
        """
        # Bag of Words単語リストを作成
        for _sentence in self.train_sentences:
            func.add_dir(_sentence)
        for _sentence in self.test_sentences:
            func.add_dir(_sentence)
        # 入力ベクトルリストを取得
        inputs = [func.bow(sentence)]

        # vectorsと同じ要素数のラベルリストを生成
        labels = [label]

        return inputs, labels

    def output(self, file_name, sentence, corr_label, pred_labels):
        """
        結果をファイルに出力する
        :param file_name: 出力ファイル
        :param sentence: 文章
        :param corr_label: 正解ラベル
        :param pred_labels: 予測ラベル
        :return: なし
        """
        # 素性がBoWなので、予測ラベルは1文につき1つだけ
        pred_label = [str(label) for label in pred_labels.data[0]]

        # 出力
        with open(file_name, 'a') as f:
            f.write(str(corr_label) + '\t' + '\t'.join(str(pred_label)) + '\t' + sentence + '\n')

        # 各文章の終わりに空行を入れる
        with open(file_name, 'a') as f:
            f.write('\n')

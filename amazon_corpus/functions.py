# -*- coding: utf-8 -*-

from dto.data import Data


def read(path: str):
    """
    Amazonコーパスを読み込む
    :param path: 読み込むAmazonコーパスのパス
    :return: List({label: 文章のラベル, sentence: 文章})
    """

    # 返すデータリスト
    data: list = []

    with open(path, 'r') as f:
        for line in f:
            # 1行ずつパースして追加
            items: list = line.split('\t')
            label: int = int(items[0]) - 1     # Amazonコーパスのラベルは1〜5だけど、学習時は0〜4にしたい
            sentence: str = items[1].rstrip()  # タブ・改行を削除
            data.append(Data(label, sentence, []))

    return data


def read_amazon_lstm_w2v(file_name: str):
    """
    AmazonコーパスのLSTMによる文章ベクトルを読み込む
    :param file_name: LSTM-Word2Vecのファイル
    :return: (文章リスト, ベクトルリスト, ラベルリスト)
    """
    sentences: list = []
    vectors: list = []
    labels: list = []
    with open(file_name, 'r') as f:
        for line in f:
            # 1行分パース
            line = line.replace('\n', '')
            items = line.split('\t')
            if len(items) == 1:
                vec = [float(v) for v in items[0].split(' ')]
                vectors.append(vec)
            # 正解ラベル/文が記述されている行
            elif len(items) == 2:
                sentences.append(items[1])
                labels.append(int(items[0]))

    return sentences, vectors, labels

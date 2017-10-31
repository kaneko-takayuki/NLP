# -*- coding: utf-8 -*-

import os
import sys
from gensim.models.keyedvectors import KeyedVectors
from econvertor.word2vec import functions


def load_w2v(file_name):
    """
    word2vecのモデルを読み込む
    :param file_name: モデルファイル
    :return なし
    """
    # word2vecのモデル読み取り
    sys.stdout.write("word2vecのモデルを読み取っています...")
    sys.stdout.flush()
    model = KeyedVectors.load_word2vec_format(file_name, binary=True)
    vector_size = model.vector_size
    print("完了")


def set_completion_func(_completion_func):
    """
    補完関数をセットする
    :param _completion_func: 補完関数
    :return: なし
    """
    completion_func = _completion_func


# パッケージ変数
model = None
vector_size = 0
set_completion_func(functions.create_zero_vector)

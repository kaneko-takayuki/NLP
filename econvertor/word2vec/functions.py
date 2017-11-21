# -*- coding: utf-8 -*-

import sys
import random
from gensim.models.keyedvectors import KeyedVectors
from econvertor import word2vec as w2v


def create_zero_vector(vector_size):
    """
    0ベクトルを作成する
    :param vector_size: ベクトルサイズ
    :return: 0ベクトル
    """
    return [0 for _ in range(vector_size)]


def create_random_vector(vector_size):
    """
    -0.25〜0.25の範囲でランダムベクトルを作成する
    :param vector_size: ベクトルサイズ
    :return: ランダムベクトル
    """
    return [random.uniform(-0.25, 0.25) for _ in range(vector_size)]


def return_none(vector_size):
    """
    Noneを返す
    :param vector_size: ベクトルサイズ
    :return: None
    """
    return None


def load_w2v(file_name):
    """
    word2vecのモデルを読み込む
    :param file_name: モデルファイル
    :return なし
    """
    # word2vecのモデル読み取り
    sys.stdout.write("word2vecのモデルを読み取っています...")
    sys.stdout.flush()
    w2v.model = KeyedVectors.load_word2vec_format(file_name, binary=True)
    w2v.vector_size = w2v.model.vector_size
    print("完了")


def set_completion_func(_completion_func):
    """
    補完関数をセットする
    :param _completion_func: 補完関数
    :return: なし
    """
    w2v.completion_func = _completion_func

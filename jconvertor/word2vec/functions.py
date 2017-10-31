# -*- coding: utf-8 -*-

import random


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

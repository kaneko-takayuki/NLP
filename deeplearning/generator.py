# -*- coding: utf-8 -*-

import random
from dto.data import Data
from dto.data import LearningData


def random_vector(d: int=200, e_min: int=0, e_max: int=0):
    """
    ランダムなベクトルを返す
    :param d: 次元数
    :param e_min: 要素の最小値
    :param e_max: 要素の最大値
    :return: 次元数dのランダムベクトル
    """
    return [random.uniform(e_min, e_max) for _ in range(d)]


def __create_flat_vector(phrases: list):
    """
    フレーズリストから、対応する1次元のベクトルを作る
    :param phrases: フレーズリスト
    :return: 入力で受け取ったベクトルリストを1次元に平坦化したもの
    """
    flat_vector: list = []
    for phrase in phrases:
        flat_vector.extend(phrase.vector)   # 単語ベクトル

    return flat_vector


def generate_with_slice(windowsize: int, completion_func, data: Data):
    """
    data.label、data.phrasesのvectorを参照して、それを対応させたLearningDataリストを生成
    :param data: Data型
    :param windowsize: ウィンドウサイズ
    :param completion_func: 補間関数
    :return: list(LearningData)
    """
    learning_data = []
    label = data.label
    phrase_n = len(data.phrases)

    if phrase_n < windowsize:  # 単語数 < 1フレーズ中の単語数
        # 表層形ベクトルリストを1次元化 -> 足りない分は補完関数で補う
        flat_vector = __create_flat_vector(data.phrases)
        for _ in range(windowsize - phrase_n):
            flat_vector.extend(completion_func)
        learning_data.append(LearningData(label, flat_vector))
    else:  # 単語数 >= 1フレーズ中の単語数
        # Word型リストから可能な限りPhrase型を生成し、リストに追加していく
        for i in range(phrase_n - windowsize + 1):
            flat_vector = __create_flat_vector(data.phrases[i:(i + windowsize)])
            learning_data.append(LearningData(label, flat_vector))

    return learning_data

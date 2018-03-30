# -*- coding: utf-8 -*-

from econvertor import word2vec


def set_completion_func(func):
    """
    :param func: word2vecでKeyErrorが発生した場合の補完コールバック関数 <class: 'function'>
    :return: なし
    """
    word2vec.completion_func = func

# -*- coding: utf-8 -*-

from analyzer.summarizer import consult_function

# デフォルトで多数決設定
consult_func = consult_function.ffnn_consult_majority


def set_consult_func(func):
    """
    合議関数を設定
    :param func: 合議関数
    :return: なし
    """
    consult_func = func

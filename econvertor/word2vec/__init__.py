# -*- coding: utf-8 -*-

import os
import sys
from gensim.models.keyedvectors import KeyedVectors
import functions

# word2vecのモデル読み取り
sys.stdout.write("word2vecのモデルを読み取っています...")
sys.stdout.flush()
base = os.path.dirname(os.path.abspath(__file__))
model = KeyedVectors.load_word2vec_format(base + "/model/GoogleNews-vectors-negative300.bin", binary=True)
vector_size = model.vector_size
print("完了")


# 補完関数の定義
completion_func = functions.create_zero_vector


def set_completion_func(_completion_func):
    """
    補完関数をセットする
    :param _completion_func: 補完関数
    :return: なし
    """
    completion_func = _completion_func

# -*- coding: utf-8 -*-

import os
import sys
from gensim.models import word2vec
from jconvertor.word2vec import functions


def load_w2v(file_name):
    """
    word2vecのモデルを読み込む
    :param file_name: 
    :return: 
    """
    # word2vecのモデル読み取り
    sys.stdout.write("word2vecのモデルを読み取っています...")
    sys.stdout.flush()
    model = word2vec.Word2Vec.load(file_name)
    vector_sizes = model.vector_size
    print("完了")


def set_completion_func(_completion_func):
    completion_func = _completion_func


# パッケージ変数
model = None
vector_size = 0
set_completion_func(functions.create_zero_vector)


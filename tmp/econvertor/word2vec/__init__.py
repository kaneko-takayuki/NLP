# -*- coding: utf-8 -*-

from econvertor.word2vec import functions


# パッケージ変数
model = None
vector_size = 0

# デフォルトで補完関数を0埋め方式にする
functions.set_completion_func(functions.create_zero_vector)

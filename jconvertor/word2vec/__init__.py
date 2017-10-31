# -*- coding: utf-8 -*-

import os
import sys
from gensim.models import word2vec
import functions

# word2vecのモデル読み取り
sys.stdout.write("word2vecのモデルを読み取っています...")
sys.stdout.flush()
base = os.path.dirname(os.path.abspath(__file__))
model = word2vec.Word2Vec.load(base + "/model/jawiki_w2v_model.bin")
vector_sizes = model.vector_size
print("完了")


def set_completion_func(_completion_func):
    completion_func = _completion_func


# 補完関数の定義
set_completion_func(functions.create_zero_vector)


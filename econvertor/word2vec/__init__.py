# -*- coding: utf-8 -*-

import os
import sys
from gensim.models.keyedvectors import KeyedVectors

# word2vecのモデル読み取り
sys.stdout.write("word2vecのモデルを読み取っています...")
sys.stdout.flush()
base = os.path.dirname(os.path.abspath(__file__))
model = KeyedVectors.load_word2vec_format(base + "/model/GoogleNews-vectors-negative300.bin", binary=True)
print("完了")


def default_callback():
    return [0 for _ in range(300)]

# 補完関数の定義
completion_func = default_callback

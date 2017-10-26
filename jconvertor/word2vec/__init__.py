# -*- coding: utf-8 -*-

import os
import sys
from gensim.models import word2vec

# word2vecのモデル読み取り
sys.stdout.write("word2vecのモデルを読み取っています...")
sys.stdout.flush()
base = os.path.dirname(os.path.abspath(__file__))
model = word2vec.Word2Vec.load(base + "/model/jawiki_w2v_model.bin")
print("完了")


def default_callback():
    return [0 for _ in range(300)]

# 補完関数の定義
completion_func = default_callback

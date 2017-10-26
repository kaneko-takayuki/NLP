# -*- coding: utf-8 -*-

import sys
import MeCab

sys.stdout.write("MeCabのモデルを読み取っています...")
sys.stdout.flush()
tagger = MeCab.Tagger("-Ochasen")
print("完了")

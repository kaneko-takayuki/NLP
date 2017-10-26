# -*- coding: utf-8 -*-

import sys
import treetaggerwrapper

sys.stdout.write("treetaggerwrapper準備中...")
sys.stdout.flush()
tagger = treetaggerwrapper.TreeTagger(TAGLANG='en', TAGDIR='/home/kaneko-takayuki/tree-tagger')
print("完了")

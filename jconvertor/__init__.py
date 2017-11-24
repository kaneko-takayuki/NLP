# -*- coding: utf-8 -*-

from jconvertor import mecab

# 日単語と英ベクトルの対応辞書
jword_to_evec = {}
j_word_keys = []
print("jconvertor: 準備中...")
with open("/home/kaneko-takayuki/NLP/ja_eng_lib/dictionary.txt", 'r') as f:
    for line in f:
        items = line.replace('\n', '').split('\t')
        jword_to_evec[items[0]] = [float(v) for v in items[1].split(',')]
        j_word_keys.append(items[0])

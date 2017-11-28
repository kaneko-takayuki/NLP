# -*- coding: utf-8 -*-

import numpy as np

from econvertor import word2vec as w2v
from econvertor.word2vec import functions as ew2v_func
from jconvertor import vectorizer


def main():

    # 実験で使用するword2vecモデルを読み込む
    ew2v_func.load_w2v("/home/kaneko-takayuki/NLP/w2v_model/GoogleNews-vectors-negative300.bin")

    # 既知語について
    with open("/home/kaneko-takayuki/NLP/ja_eng_lib/similarity_test/well_known_words_corpus.txt", 'r') as f:
        with open("/home/kaneko-takayuki/NLP/ja_eng_lib/similarity_test/well_known_words_result.txt", 'w') as o:
            for line in f:
                word = line.replace('\n', '')
                result = w2v.model.most_similar(positive=[np.asarray(vectorizer.word_vector_to_eng(word))], topn=10)
                o.write(word + ": " + str(result) + '\n')

    # 未知語について
    with open("/home/kaneko-takayuki/NLP/ja_eng_lib/similarity_test/unknown_words_corpus.txt", 'r') as f:
        with open("/home/kaneko-takayuki/NLP/ja_eng_lib/similarity_test/unknown_words_result.txt", 'w') as o:
            for line in f:
                word = line.replace('\n', '')
                result = w2v.model.most_similar(positive=[np.asarray(vectorizer.word_vector_to_eng(word))], topn=10)
                o.write(word + ": " + str(result) + '\n')

if __name__ == '__main__':
    main()

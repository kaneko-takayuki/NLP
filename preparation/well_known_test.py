# -*- coding: utf-8 -*-

from jconvertor import word2vec as w2v
from jconvertor.word2vec import functions as w2v_func
from jconvertor import spliter
from amazon_corpus import functions


def main():
    """
    既知語と未知語のタイプ数を調べる
    :return: なし
    """
    well_known_n = 0
    unknown_n = 0

    # word2vec読み込み
    w2v_func.load_w2v("/home/kaneko-takayuki/NLP/w2v_model/nwcj_word_1_200_8_25_0_1e4_32_1_15.bin")

    # Amazonコーパスを読み取る
    data = functions.read_amazon_corpus("/home/kaneko-takayuki/NLP/amazon_corpus/data/jp/books/all.tsv")
    for sentence in data[0]:
        words = spliter.words(sentence)
        for word in words:
            try:
                _ = w2v.model[word]
                well_known_n += 1
            except KeyError:
                unknown_n += 1

    print("既知語: " + str(well_known_n))
    print("未知語: " + str(unknown_n))

if __name__ == '__main__':
    main()

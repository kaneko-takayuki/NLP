# -*- coding: utf-8 -*-

from amazon_corpus import functions
from jconvertor import spliter
from ja_eng_lib.functions import read_all


def main():
    corpus_data = functions.read_amazon_corpus("/home/kaneko-takayuki/NLP/amazon_corpus/data/jp/books/all.tsv")
    word_set = set()
    well_known_words = read_all().keys()

    for sentence in corpus_data[0]:
        words = spliter.words(sentence)
        for word in words:
            word_set.add(word)

    with open("/home/kaneko-takayuki/NLP/ja_eng_lib/similarity_test/well_known_words_corpus.txt", 'w') as f1:
        with open("/home/kaneko-takayuki/NLP/ja_eng_lib/similarity_test/unknown_words_corpus.txt", 'w') as f2:
            for word in word_set:
                if word in well_known_words:
                    f1.write(word + '\n')
                else:
                    f2.write(word + '\n')

if __name__ == '__main__':
    main()

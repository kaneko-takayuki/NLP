# -*- coding: utf-8 -*-

import add_path
from amazon_corpus import functions
from jconvertor import spliter


def main():
    corpus_data = functions.read_amazon_corpus("/home/kaneko-takayuki/NLP/amazon_corpus/data/jp/books/processed_test.review")
    word_set = set()

    for sentence in corpus_data[0]:
        words = spliter.words(sentence)
        for word in words:
            word_set.add(word)

    with open("/home/kaneko-takayuki/NLP/ja_eng_lib/similarity_test/word_corpus.txt", 'w') as f:
        f.write('\n'.join(word_set))

if __name__ == '__main__':
    main()

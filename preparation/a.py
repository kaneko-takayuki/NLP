# -*- coding: utf-8 -*-

import add_path
from ja_eng_lib.functions import read_all


def main():
    well_known_words = read_all().keys()

    with open("/home/kaneko-takayuki/NLP/ja_eng_lib/similarity_test/method2/test_review_words_result.txt", 'r') as i:
        for line in i:
            word = line.split(': ')[0]
            if word in well_known_words:
                with open("/home/kaneko-takayuki/NLP/ja_eng_lib/similarity_test/method2/well_known_words_result.txt", 'a') as o:
                    o.write(line)
            else:
                with open("/home/kaneko-takayuki/NLP/ja_eng_lib/similarity_test/method2/unknown_words_result.txt", 'a') as o:
                    o.write(line)

if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-

import sys
import MeCab

import SentenceConvertor

# MeCab、word2vecを使用する準備
sys.stdout.write("BagofWords: MeCab準備中...")
tagger = MeCab.Tagger("-Ochasen")
print("完了")


class BagofWords:
    """
    文章からBag of Wordsを作成するクラス
    """

    def __init__(self):
        self.words_dir = {}

    def add_dir(self, sentence=""):
        """
        文を与えて、単語辞書を追加する
        :param sentence: 追加する文章
        :return: なし
        """
        s = SentenceConvertor.SentenceConvertor(sentence)

        for word in s.get_original_wakati().split():
            if word not in self.words_dir:
                n_words = len(self.words_dir)
                self.words_dir[word] = n_words

    def get_bow(self, sentence):
        """
        Bag of Wordsを返す
        :param sentence: Bag of Wordsにする文
        :return: bag of wordsのリスト
        """
        # 総単語数と同じ長さのリストを0埋めで作成
        bag_of_words = [0 for _ in range(len(self.words_dir))]
        s = SentenceConvertor.SentenceConvertor(sentence)

        # 単語を見て、共起する場所のみ1を立てる
        for word in s.get_original_wakati().split():
            bag_of_words[self.words_dir[word]] = 1

        return bag_of_words

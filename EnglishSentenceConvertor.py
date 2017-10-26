# -*- coding: utf-8 -*-

import sys
import random
from gensim.models.keyedvectors import KeyedVectors

import treetaggerwrapper

import Constant

# treetaggerwrapper, word2vecを使用する準備
sys.stdout.write("EnglishSentenceConvertor: treetaggerwrapper準備中...")
tagger = treetaggerwrapper.TreeTagger(TAGLANG='en', TAGDIR='/home/kaneko-takayuki/tree-mecab')
print("完了")
sys.stdout.write("EnglishSentenceConvertor: word2vec準備中...")
w2v_model = KeyedVectors.load_word2vec_format("model/GoogleNews-vectors-negative300.bin", binary=True)
print("完了")


class EnglishSentenceConvertor:
    """
    文から様々な形式に変換を行うクラス
    """

    def __init__(self, sentence=""):
        """
        :param sentence: 変換対象となる文
        """
        # 文の形態素解析結果を保持しておく
        # 『単語, 品詞, 原型』の情報が parsed_sentence に格納される
        self.sentence = sentence

        tags = tagger.TagText(sentence)
        self.parsed_sentence = []
        for tag in tags:
            self.parsed_sentence.append(tag.split())

    def __call__(self):
        """
        :return: 元の文
        """
        return self.sentence

    def get_word_count(self):
        """
        :return: 文の単語数
        """
        return len(self.get_wakati().split())

    def get_wakati(self, window_size=0):
        """
        :return: 文の分かち書き
        """
        words = [x[0] for x in self.parsed_sentence]
        if (window_size == 0):
            return " ".join(words)

        if len(words) <= window_size:
            return [" ".join(words)]

        # window_sizeが指定されている時は、そのサイズで切り出す
        wakati = []
        for i in range(len(words) - window_size + 1):
            wakati.append(" ".join(words[i:i+window_size]))
        return wakati

    def get_original_wakati(self):
        """
        :return: 単語を全て原型に変換した分かち書き
        """
        return ' '.join([x[2] for x in self.parsed_sentence])

    def get_vectors(self, window_size=0, completion_method=0):
        """
        Word2Vecによって各単語を分散表現に変換して返す
        window_sizeを指定した場合、そのサイズで単語を区切って返す
        :param window_size: 分割する場合のウィンドウサイズ
        :param completion_method: KeyError等が起きた際の補完方法(0:-0.25〜0.25のランダム値で補完, 1:0埋めしたもので補完)
        :return: 分散表現のリスト
        """
        # 分かち書きを取得
        wakati = self.get_wakati()

        # それぞれの単語に対して、Word2Vecから分散表現を求める
        # 未知の単語の場合は-0.25〜0.25のランダム値で埋める
        vectors = []
        for word in wakati.split():
            try:
                vectors.append(w2v_model[word])
            except KeyError:
                if completion_method == 0:
                    vectors.append([random.uniform(-0.25, 0.25) for _ in range(300)])
                else:
                    vectors.append([0.0 for _ in range(300)])

        # 分割しない場合はそのまま返す
        if window_size == 0:
            return vectors

        # 単語数が少な過ぎて、指定されたウィンドウサイズで分割できない時は、末尾にランダムな分散表現を付与し、
        # 最低でも一つ以上の分散表現を得られるようにする
        while len(vectors) < window_size:
            if completion_method == 0:
                vectors.append([random.uniform(-0.25, 0.25) for _ in range(300)])
            else:
                vectors.append([0.0 for _ in range(300)])

        # 分散表現を求める
        # split_vectors: 返す素性のリスト
        split_vectors = []
        for i in range(0, len(vectors)-window_size+1):
            append_vector = []
            # 単語分散表現を取り出し、append_vectorに埋め込んでいく
            for word_vector in vectors[i:i+window_size]:
                append_vector.extend(word_vector)
            split_vectors.append(append_vector)

        return split_vectors

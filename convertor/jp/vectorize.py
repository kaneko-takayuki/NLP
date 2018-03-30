# -*- coding: utf-8 -*-

import random
from gensim.models import KeyedVectors
from convertor.jp.morphological_analyse import extract_original
import constants
from dto.data import Word, Phrase
from log.functions import create_logger

# ロガー
logger = create_logger(__name__)


def __create_phrase(words: list):
    """
    WordDataリストからPhraseDataを生成する
    :param words: WordDataリスト
    :return: PhraseData
    """
    word_list: list = []
    vector: list = []
    for word in words:
        word_list.append(word.word)  # 単語
        vector.extend(word.vector)   # 単語ベクトル

    return Phrase(' '.join(word_list), vector)


def random_vector(d: int=200, e_min: int=0, e_max: int=0):
    """
    ランダムなベクトルを返す
    :param d: 次元数
    :param e_min: 要素の最小値
    :param e_max: 要素の最大値
    :return: 次元数dのランダムベクトル
    """
    return [random.uniform(e_min, e_max) for _ in range(d)]


def load_model(w2v_dic_path: str):
    """
    word2vec用辞書パスを渡して、読み込んだモデルを返す
    :param w2v_dic_path: word2vec辞書パス
    :return: モデル
    """
    return KeyedVectors.load_word2vec_format(w2v_dic_path, binary=True)

# nwjc2vecを読み込む
logger.info('word2vecファイルを読み込んでいます。')
w2v_model = load_model(constants.project + '/convertor/jp/nwjc_word_1_200_8_25_0_1e4_32_1_15.bin')
logger.info('word2vecファイルの読み込みが完了しました。')


def give_vector(word: str, completion_func=random_vector(200, 0, 0)):
    """
    単語をベクトル化する
    :param word: 単語
    :param completion_func: wordが未知語の時に代わりに返すベクトルを生成する補完関数
    :return: Word(単語, 単語ベクトル)
    """
    if w2v_model.__contains__(word):
        return Word(word, w2v_model[word])
    else:
        return Word(word, completion_func)


def create_phrases(sentence: str, windowsize: int=3, completion_func=random_vector(200, 0, 0)):
    """
    文章をフレーズベクトルリストに変換する
    :param sentence: 文章
    :param windowsize: 1フレーズ中の単語数
    :param completion_func: 文章中の単語数が足りない時、補完を行う関数
    :return: フレーズベクトルリスト
    """
    # 文章 -> 表層形リスト -> 表層形ベクトルリスト
    original_words: iter = extract_original(sentence)
    words: list = list(map(lambda word: give_vector(word), original_words))
    words_n: int = len(words)  # 抜き出せた単語数

    phrases: list = []  # フレーズベクトルリスト

    if words_n < windowsize:  # 単語数 < 1フレーズ中の単語数
        # 表層形ベクトルリストを1次元化 -> 足りない分は補完関数で補う
        phrase = __create_phrase(words)
        for _ in range(windowsize - words_n):
            phrase.vector.extend(completion_func)
        phrases.append(phrase)
    else:  # 単語数 >= 1フレーズ中の単語数
        # Word型リストから可能な限りPhrase型を生成し、リストに追加していく
        for i in range(words_n - windowsize + 1):
            phrases.append(__create_phrase(words[i:(i + windowsize)]))

    return iter(phrases)

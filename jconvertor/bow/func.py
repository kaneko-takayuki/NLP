# -*- coding: utf-8 -*-

from jconvertor import spliter
from jconvertor.bow import words_dir


def add_dir(sentence=""):
    """
    単語辞書を追加する
    :param sentence: 日本語文章 <class: 'str'>
    :return: なし
    """
    original_words = spliter.original_words(sentence)
    # 新しい単語が現れたら、辞書に追加する
    for word in original_words:
        if word not in words_dir:
            _dir_len = dir_len()
            words_dir[word] = _dir_len


def dir_len():
    """
    辞書に登録済の単語数を返す
    :return: 登録済の単語数
    """
    return len(words_dir)


def bow(sentence):
    """
    辞書と照らし合わせてBag of Wordsを返す
    :param sentence: 日本語文章 <class: 'str'>
    :return: Bag of Wordsのリスト
    """
    # 単語の登録数の長さのリストを0埋めで作成
    _bow = [0 for _ in range(dir_len())]

    # 単語を見て、共起する場所のみ1を立てる
    original_words = spliter.original_words(sentence)
    for word in original_words:
        _bow[words_dir[word]] = 1

    return _bow

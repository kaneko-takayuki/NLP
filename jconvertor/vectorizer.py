# -*- coding: utf-8 -*-

import sys
from jconvertor import spliter
from jconvertor import word2vec as w2v


def word_vector(word):
    """
    単語をベクトル化する
    :param word: 単語 <class: 'str'>
    :return: 単語ベクトル <class: 'list'>
    """
    try:
        return w2v.model[word]
    except KeyError:
        # 引数として、辞書に存在しない単語が渡された
        if w2v.completion_func is None:
            sys.stderr.write("KeyError: 補完用のコールバック関数を指定してください。\n")
            exit()
        return w2v.completion_func()
    except TypeError:
        # 引数に文字型以外が渡された
        sys.stderr.write("TypeError: 引数の型を見なおしてください。\n")
        exit()


def phrase_vector(phrase):
    """
    フレーズをベクトル化する
    :param phrase: フレーズ(単語リスト) <class: 'list'>
    :return: フレーズベクトル <class: 'list'>
    """
    vector = []
    # それぞれの単語について、ベクトルを取得して連結する
    for word in phrase:
        vector.extend(word_vector(word))
    return vector


def sentence_vector(sentence, window_size=1):
    """
    文章をベクトル化する
    :param sentence: 日本語文章 <class: 'str'>
    :param window_size: 切り取るウィンドウサイズ <class: 'int'> 
    :return: 文章から得られるフレーズベクトルリスト <class: 'list'>
    """
    _sentence_vector = []
    count = spliter.word_count(sentence)
    phrases = spliter.phrases(sentence, window_size)

    if len(phrases) == 0:
        # window_sizeが単語数より大きくて、フレーズが得られなかった
        # 補完関数を用いて、window_sizeに合うようにベクトルを生成
        completion_vector = []
        words = spliter.words(sentence)
        # それぞれの単語についてベクトルを取得
        for word in words:
            completion_vector.extend(word_vector(word))
        # 補完関数が設定されていなかったら、エラー処理
        if w2v.completion_func is None:
            sys.stderr.write("KeyError: 補完用のコールバック関数を指定してください。\n")
            exit()
        # 補完を行う
        for _ in range(window_size - count):
            completion_vector.extend(w2v.completion_func())
        # 補完済のベクトルをフレーズベクトルリストとする
        _sentence_vector.append(completion_vector)
    else:
        # フレーズを一つベクトル化する
        for phrase in phrases:
            _sentence_vector.append(phrase_vector(phrase))

    return _sentence_vector


if __name__ == '__main__':
    test_s = "単語"
    print(word_vector(test_s))
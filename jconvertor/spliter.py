# -*- coding: utf-8 -*-

from jconvertor import mecab


def words(sentence):
    """
    単語分割を行う
    :param sentence: 日本語文章 <class: 'str'>
    :return: 単語リスト <class: 'list'>
    """
    # 受け取った文章を形態素解析
    # MeCabの仕様上、最後2つ分にEOSと空行が入るので、それは省く
    parsed_s = mecab.tagger.parse(sentence)

    parsed_s = parsed_s.split('\n')
    parsed_s = parsed_s[:-2]

    # パース済のデータに対して、1番目は単語そのまま出てくる
    return [x.split('\t')[0] for x in parsed_s]


def original_words(sentence):
    """
    単語分割を行い、原型変換を行う
    :param sentence: 日本語文章 <class: 'str'>
    :return: 原型単語リスト <class: 'list'>
    """
    # 受け取った文章を形態素解析
    # MeCabの仕様上、最後2つ分にEOSと空行が入るので、それは省く
    parsed_s = mecab.tagger.parse(sentence).split('\n')[:-2]

    # パース済のデータに対して、3番目は原型が出てくる
    return [x.split('\t')[2] for x in parsed_s]


def phrases(sentence, window_size=1):
    """
    文章に対するフレーズを得る
    :param sentence: 日本語文章 <class: 'str'>
    :param window_size: フレーズの分割単位 <class: 'int'>
    :return: フレーズリスト <class: 'list'>
    """
    _phrases = []
    _words = words(sentence)
    _count = word_count(sentence)

    for i in range(_count - window_size + 1):
        # i番目からwindow_size個分の単語リストを取得する
        _phrases.append(_words[i:i+window_size])

    return _phrases


def wakati(sentence):
    """
    分かち書きを行う
    :param sentence: 日本語文章 <class: 'str'>
    :return: 分かち書きした結果 <class: 'str'>
    """
    return ' '.join(words(sentence))


def word_count(sentence):
    """
    文中の単語数を数える
    :param sentence: 日本語文章 <class: 'str'>
    :return: 単語数 <class: 'int'>
    """
    return len(words(sentence))

if __name__ == '__main__':
    test_s = "今日はいい天気と思いたい"
    print(original_words(test_s))

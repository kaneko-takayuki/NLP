# -*- coding: utf-8 -*-

"""
形態素解析関連の関数をまとめたファイル
"""

import treetaggerwrapper as ttw
from dto.data import ParseResult

# 辞書の読み込み
tagger = ttw.TreeTagger(TAGLANG='en', TAGDIR='/home/kaneko-takayuki/tree-tagger')


def __convert(x: list):
    """
    パースした結果をParseData型に変換する
    :param x: パースした結果
    :return: ParseData型
    """
    surface: str = x[0] if len(x) > 0 else ''
    original: str = x[1] if len(x) > 1 else ''
    return ParseResult(surface, original)


def parse(sentence: str):
    """
    文章に対して形態素解析を実行する
    :param sentence: 文章
    :return: 形態素解析リスト

    parse('I was born')
      -> List('I'の解析結果, 'was'の解析結果, 'born'の解析結果)
      ※ 「解析結果」とは、PhraseResult型である
    """
    # TreeTaggerで形態素解析
    # 表層形 \t 品詞 \t 原形 の順で出力されてくるので、タブで区切っておく
    tmp_parse_result = tagger.TagText(sentence)
    tmp_parse_result = map(lambda x: x.split('\t'), tmp_parse_result)
    # それぞれParseResult型に変換
    parse_result = map(__convert, tmp_parse_result)

    return parse_result


def extract_surface(sentence: str):
    """
    文章に対して、形態素解析して表層形を抜き出して返す
    :param sentence: 文章
    :return: 表層形リスト

    extract_surface('I was born')
      -> List('I', 'was', 'born') 
    """
    parse_results = parse(sentence)
    return map(lambda result: result.surface, parse_results)


def extract_original(sentence: str):
    """
    文章に対して、形態素解析をして原形を抜き出して返す
    :param sentence: 文章
    :return: 原形リスト

    extract_original('I was born')
      -> List('I', 'be', 'bear')
    """
    parse_results = parse(sentence)
    return map(lambda result: result.original, parse_results)


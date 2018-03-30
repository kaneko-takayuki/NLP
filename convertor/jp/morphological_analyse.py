# -*- coding: utf-8 -*-

"""
形態素解析関連の関数をまとめたファイル
出力形式は、以下を想定(動作確認はunidicで行った)

【出力形式】
---------------------------------------
output-format-type = unidic

node-format-unidic = %m\t%f[9]\t%f[6]\t%f[7]\t%F-[0,1,2,3]\t%f[4]\t%f[5]\n
unk-format-unidic  = %m\t%m\t%m\t%m\t%F-[0,1,2,3]\t%f[4]\t%f[5]\n
bos-format-unidic  =
eos-format-unidic  = EOS\n

node-format-chamame = \t%m\t%f[9]\t%f[6]\t%f[7]\t%F-[0,1,2,3]\t%f[4]\t%f[5]\n
;unk-format-chamame = \t%m\t\t\t%m\tUNK\t\t\n
unk-format-chamame  = \t%m\t\t\t%m\t%F-[0,1,2,3]\t\t\n
bos-format-chamame  = B
eos-format-chamame  = 
---------------------------------------
"""

import MeCab
from dto.data import ParseResult


def create_tagger(mecab_dic_path: str):
    """
    形態素解析用の辞書パスを渡して、解析ハンドラを生成して返す
    :param mecab_dic_path: 形態素解析用の辞書パス
    :return: 解析ハンドラ
    """
    return MeCab.Tagger('-d' + mecab_dic_path)


# 辞書の読み込み
tagger = create_tagger('/var/lib/mecab/dic/ipadic-utf8')


def __convert(x: dict):
    """
    パースした結果をParseData型に変換する
    :param x: パースした結果
    :return: ParseData型
    """
    surface: str = x['surface']
    original: str = x['info'][6]
    return ParseResult(surface, original)


def parse(sentence: str):
    """
    文章に対して形態素解析を実行する
    :param sentence: 文章
    :return: 形態素解析リスト

    parse('ケーキ食べたかった')
      -> List('ケーキ'の解析結果, '食べる'の解析結果, 'たい'の解析結果, 'た'の解析結果)
      ※「解析結果」とは、PhraseResult型である
    """
    # MeCabでパースすると、末尾にEOSと空行が入るので、後ろ2つは切り捨てて、
    # すると'表層形\t形態素解析情報'の順で出てくるので、そこから原型と品詞だけ抜き出している
    tmp_parse_result = tagger.parse(sentence).split('\n')[:-2]
    tmp_parse_result = map(lambda x: {
        'surface': x.split('\t')[0],
        'info': x.split('\t')[1].split(',')
    }, tmp_parse_result)

    # 解析結果が'info'でまとまっているので分離
    # それぞれParseResult型に変換
    parse_result = map(__convert, tmp_parse_result)

    return parse_result


def extract_surface(sentence: str):
    """
    文章に対して、形態素解析して表層形を抜き出して返す
    :param sentence: 文章
    :return: 表層形リスト

    extract_surface('ケーキ食べたかった')
      -> List('ケーキ', '食べ', 'たかっ', 'た') 
    """
    parse_results = parse(sentence)
    return map(lambda result: result.surface, parse_results)


def extract_original(sentence: str):
    """
    文章に対して、形態素解析をして原形を抜き出して返す
    :param sentence: 文章
    :return: 原形リスト

    extract_original('ケーキ食べたかった')
      -> List('ケーキ', '食べる', 'たい', 'た')
    """
    parse_results = parse(sentence)
    return map(lambda result: result.original, parse_results)


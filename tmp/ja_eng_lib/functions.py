# -*- coding: utf-8 -*-


def read_ja_to_eng():
    """
    日本単語-英単語対応辞書を読み込む
    :return: {日本語: [対応英単語1, 対応英単語2, ..., 対応英単語n]}
    """
    ja_eng_dir = {}
    with open("/home/kaneko-takayuki/NLP/ja_eng_lib/ja_to_eng.tsv", 'r') as f:
        for line in f:
            # 1行分パース
            items = line.replace('\n', '').split('\t')  # ID, 元単語, 対応単語列
            if items[1] not in ja_eng_dir:
                ja_eng_dir[items[1]] = []
            for corr_word in items[2].split(','):
                ja_eng_dir[items[1]].append(corr_word)

    return ja_eng_dir


def read_eng_to_ja():
    """
    日本単語-英単語対応辞書を読み込む
    :return: {英語: [対応日単語1, 対応日単語2, ..., 対応日単語n]}
    """
    ja_eng_dir = {}
    with open("/home/kaneko-takayuki/NLP/ja_eng_lib/eng_to_ja.tsv", 'r') as f:
        for line in f:
            # 1行分パース
            items = line.replace('\n', '').split('\t')  # ID, 元単語, 対応単語列
            if items[1] not in ja_eng_dir:
                ja_eng_dir[items[1]] = []
            for corr_word in items[2].split(','):
                ja_eng_dir[items[1]].append(corr_word)

    return ja_eng_dir


def read_all():
    """
    日本語-英語の対応を全て読み込む
    :return: 
    """
    # ファイルを読み込む
    ja_to_eng = read_ja_to_eng()
    eng_to_eng = read_eng_to_ja()

    # 英語->日本語辞書を、日本語->英語辞書に埋め込む形でマージ
    for e_word in eng_to_eng.keys():
        for j_word in eng_to_eng[e_word]:
            if j_word not in ja_to_eng:
                ja_to_eng[j_word] = []
            ja_to_eng[j_word].append(e_word)
            ja_to_eng[j_word] = list(set(ja_to_eng[j_word]))

    return ja_to_eng  # 一度set型に書き換えることで、重複を排除

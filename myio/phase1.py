# -*- coding: utf-8 -*-
from dto.data import Data, Phrase


def write(path, data):
    """
    dataの内容をpathに出力する
    :param path: 出力するファイルパス
    :param data: 出力する内容データ
    :return: なし
    """
    with open(path, 'w') as f:
        for row in data:
            # {ラベル}\t{文章}
            f.write(str(row.label) + '\t' + row.sentence + '\n')
            # 文章中の単語ベクトル(数値をstrに変換して、空白でjoin)
            for phrase in row.phrases:
                vector_str = map(str, phrase.vector)
                f.write(phrase.phrase + '\t' + ' '.join(vector_str) + '\n')
            # 文と文に間は、空行を入れる
            f.write('\n')


def parse_vector(vector_str):
    """
    ベクトルリスト文字列から、実際のfloatリストにパース
    :param vector_str: ベクトルリスト文字列
    :return: floatリスト
    """
    return list(map(float, vector_str.split()))


def read(path):
    """
    pathのファイルを読み取り、オブジェクトリストに変換する
    :param path: ファイルパス
    :return: list(Data)
    """
    data = []

    with open(path, 'r') as f:
        head_flag = True
        label = -1
        sentence = ''
        phrases = []
        for line in f:
            # 改行は除く
            line = line.rstrip()

            if len(line) == 0:  # 空行が現れたら、それは1文が終わったということ
                data.append(Data(label, sentence, phrases))
                # 1文終わったので、設定項目を初期化して次の文章へ移る
                head_flag = True
                label = -1
                sentence = ''
                phrases = []
            elif head_flag:  # 文章データの先頭(ラベルと文章)
                items = line.split('\t')
                label = int(items[0])
                sentence = items[1]
                head_flag = False
            else:  # フレーズベクトル
                items = line.split('\t')
                phrases.append(Phrase(items[0], parse_vector(items[1])))

    return data

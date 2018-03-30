# -*- coding: utf-8 -*-

import argparse
import functools
from amazon_corpus.functions import read as amazon_read
from convertor.jp2en.vectorize import create_phrases
from myio.phase1 import write
from log.functions import create_logger
from dto.data import Data

# ロガー
logger = create_logger(__name__)


def convert(windowsize: int, data: Data):
    """
    データ1行分を入力に取って、単語ベクトルリストを付与して返す
    :param windowsize: 1フレーズ中に含まれる単語数
    :param data: 1行分のデータ Data型
    :return: 単語ベクトルリストを付与したData型
    """
    # windowsize=1を指定することで、単語毎のベクトルが得られる
    phrases = create_phrases(sentence=data.sentence, windowsize=windowsize)
    return Data(data.label, data.sentence, phrases)


def main(input_file: str, output_file: str, windowsize: int):
    logger.info('----------')
    logger.info('main()')
    logger.info('input_file: ' + input_file)
    logger.info('output_file: ' + output_file)
    logger.info('windowsize: ' + str(windowsize))

    w_convert = functools.partial(convert, windowsize)

    # データを読み込んで、文章中の単語をそれぞれ単語ベクトルに変換する
    data = amazon_read(input_file)
    converted_data = map(w_convert, data)
    write(output_file, converted_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Amazonコーパスの日本語データを得英語ベクトルに変換して出力する')
    parser.add_argument('--inputs', '-i', nargs='*', default='', help='Amazonコーパスの日本語データのパス(複数指定可能)')
    parser.add_argument('--outputs', '-o', nargs='*', default='', help='ベクトル変換後、吐き出すファイルパス(複数指定可能)')
    parser.add_argument('--windowsize', '-w', type=int, default=1, help='1フレーズ中の単語数')
    args = parser.parse_args()

    # 引数のエラー処理
    if len(args.inputs) != len(args.outputs):
        logger.error('入力ファイルの数と出力ファイルの数は合わせてください')
        exit()

    for (_input_file, _output_file) in zip(args.inputs, args.outputs):
        main(_input_file, _output_file, int(args.windowsize))

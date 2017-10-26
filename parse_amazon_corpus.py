# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET

import Constant

# XML形式のamazonコーパスを整形する
def parse_xml_file(input_file, output_file):
    # xmlファイルを読み取る
    tree = ET.parse(input_file)
    items = tree.getroot()

    with open(output_file, "w") as f:
        for item in items:
            # 評価情報・レビュー文を読み取る
            rating = item[1].text.strip()
            rating = str(int(float(rating)))
            sentence = item[2].text.strip()

            # レビュー文の整形
            # 改行コード削除・空白が2つ以上並ぶ場合には1つの空白に置き換える
            sentence = sentence.replace("\n", "")
            while "  " in sentence:
                sentence = sentence.replace("  ", " ")

            # (評価)\t(文)\n
            # の形式で出力する
            output_text = rating + "\t" + sentence + "\n"
            f.write(output_text)

if __name__ == "__main__":
    base_path = Constant.BASE_PATH + "amazon_corpus/"

    genre = ['books', 'dvd', 'music']
    for genre_name in genre:
        input_path = base_path + "unprocessed_data/" + genre_name + "/test.review"
        output_path = base_path + 'data/' + genre_name + "/test.tsv"
        parse_xml_file(input_path, output_path)

        input_path = base_path + "unprocessed_data/" + genre_name + "/train.review"
        output_path = base_path + 'data/' + genre_name + "/train.tsv"
        parse_xml_file(input_path, output_path)

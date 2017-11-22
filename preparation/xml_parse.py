# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET

def main():
    tree = ET.parse('/home/kaneko-takayuki/NLP/amazon_corpus/data/jp/books/test.review')
    root = tree.getroot()

    with open("/home/kaneko-takayuki/NLP/amazon_corpus/data/jp/books/processed_test.review", 'w') as f:
        for item in root:
            f.write(str(int(float(item[1].text))) + '\t' + str(item[4].text).replace('\n', '') + '\n')

if __name__ == '__main__':
    main()

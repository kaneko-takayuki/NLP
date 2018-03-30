# -*- coding: utf-8 -*-

import os


def correct(file_name):
    space_line_n = 0
    with open(file_name, 'r') as f:
        with open(file_name + "_tmp", 'w') as ftmp:
            for line in f:
                items = line.split()
                if len(items) == 0:
                    space_line_n += 1
                    if space_line_n % 2 == 1:
                        continue
                ftmp.write(line)

    os.remove(file_name)
    os.rename(file_name + "_tmp", file_name)


def main():
    # プロジェクトディレクトリを取得
    base_path = os.path.dirname(os.path.abspath(__file__)) + "/../"
    _window = ["window3", "window5", "window5"]

    for w in _window:
        for i in range(1, 6):
            for j in range(1, 31):
                file_name = base_path + "amazon_corpus/Sigmoid_W2V/" + w + "/random/out/cross_validation1" + str(i) + \
                            "/epoch" + str(j) + ".tsv"

                correct(file_name)


if __name__ == '__main__':
    main()

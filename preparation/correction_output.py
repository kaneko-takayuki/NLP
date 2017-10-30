# -*- coding: utf-8 -*-

import os
import sys


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

    correct(base_path + "test.tsv")


if __name__ == '__main__':
    main()

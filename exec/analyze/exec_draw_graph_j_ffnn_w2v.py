# -*- coding: utf-8 -*-

import os
import argparse
import add_path
from analyzer.graph_drawer import drawer


def main(x, input_files, labels):
    # プロジェクトディレクトリを取得
    base_path = os.path.dirname(os.path.abspath(__file__)) + "/../../"

    # x軸の長さリスト
    left = [i for i in range(1, x+1)]

    # グラフを描画
    drawer.draw_graph(left, input_files, labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FFNN_parameter')
    parser.add_argument("--x", "-x", type=int)
    parser.add_argument("--files", "-f", nargs='*')
    parser.add_argument("--labels", "-l", nargs='*')
    args = parser.parse_args()

    main(args.x, args.files, args.labels)

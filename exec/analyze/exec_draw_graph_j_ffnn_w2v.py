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
    parser = argparse.ArgumentParser(description='折れ線グラフを描画する')
    parser.add_argument("--x", "-x", type=int, help='横軸の長さ')
    parser.add_argument("--files", "-f", nargs='*', help='グラフに表示する精度ファイル(複数指定可能)')
    parser.add_argument("--labels", "-l", nargs='*', help='線の色')
    args = parser.parse_args()

    main(args.x, args.files, args.labels)

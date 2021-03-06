# -*- coding: utf-8 -*-

import os
import argparse
from analyzer.graph_drawer import drawer


def main(x, input_files, labels, xlabel, ylabel, title, min_ylim, max_ylim):
    # プロジェクトディレクトリを取得
    base_path = os.path.dirname(os.path.abspath(__file__)) + "/../../"

    # x軸の長さリスト
    left = [i for i in range(1, x+1)]

    # グラフを描画
    drawer.draw_graph_files(left, input_files, labels, xlabel, ylabel, title, min_ylim, max_ylim)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='折れ線グラフを描画する')
    parser.add_argument("--x", "-x", type=int, help='横軸の長さ')
    parser.add_argument("--files", "-f", nargs='*', help='グラフに表示する精度ファイル(複数指定可能)')
    parser.add_argument("--labels", "-l", nargs='*', help='線の色')
    parser.add_argument("--xlabel", "-xl", help='横軸の説明')
    parser.add_argument("--ylabel", "-yl", help='縦軸の説明')
    parser.add_argument("--min_ylim", "-miny", type=float, help='縦軸の最小値')
    parser.add_argument("--max_ylim", "-maxy", type=float, help='横軸の最小値')
    parser.add_argument("--title", "-t", help='グラフのタイトル')
    args = parser.parse_args()

    main(args.x, args.files, args.labels, args.xlabel, args.ylabel, args.title, args.min_ylim, args.max_ylim)

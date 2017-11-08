# -*- coding: utf-8 -*-

import argparse
import add_path
from analyzer.graph_drawer import drawer


def main(x, input_file, rows, labels, xlabel, ylabel, title, min_ylim, max_ylim):
    # x軸の長さリスト
    left = [i for i in range(1, x + 1)]

    # グラフを描画
    drawer.draw_graph_file(left, input_file, rows, labels, xlabel, ylabel, title, min_ylim, max_ylim)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='一つのファイルに格納されているラベル毎の出力値グラフを描画')
    parser.add_argument("--x", "-x", type=int, help='横軸の長さ')
    parser.add_argument("--file", "-f", help='グラフに表示する精度ファイル')
    parser.add_argument("--rows", "-r", type=int, nargs='*', help='ファイルの何列目に対するグラフを表示するか')
    parser.add_argument("--labels", "-l", nargs='*', help='グラフラベル')
    parser.add_argument("--xlabel", "-xl", help='横軸の説明')
    parser.add_argument("--ylabel", "-yl", help='縦軸の説明')
    parser.add_argument("--min_ylim", "-miny", type=float, help='縦軸の最小値')
    parser.add_argument("--max_ylim", "-maxy", type=float, help='横軸の最小値')
    parser.add_argument("--title", "-t", help='グラフのタイトル')
    args = parser.parse_args()

    main(args.x, args.file, args.rows, args.labels, args.xlabel, args.ylabel, args.title, args.min_ylim, args.max_ylim)

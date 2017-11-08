# -*- coding: utf-8 -*-

"""
5値分類で得たoutファイルについて、
小数点第5桁に揃えて出力する
"""


def main(input_file, output_file):
    with open(input_file, 'r') as i, open(output_file, 'w') as o:
        for line in i:
            line = line.replace('\n', '')
            items = line.split('\t')
            if len(items) != 5:
                o.write(line + '\n')
            else:
                shaped_items = [str(round(float(pred), 12)).ljust(15, ' ') for pred in items[:4]]
                o.write('\t'.join(shaped_items) + '\t' + items[4] + '\n')

if __name__ == '__main__':
    main('out_epoch50.tsv', 'shaped_out_epoch50.tsv')

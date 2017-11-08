# -*- coding: utf-8 -*-


def main(input_file, output_file):
    with open(input_file, 'r') as i, open(output_file, 'w') as o:
        correct_items = []
        phrase_items = []
        for line in i:
            line = line.replace('\n', '')
            items = line.split('\t')
            if len(items) == 2:
                correct_items = items
                continue
            elif len(items) == 7:
                phrase_items.append(items)
                continue
            else:
                label_n = [0 for _ in range(5)]
                for items in phrase_items:
                    label_n[int(items[0])] += 1
                max_n = 0
                label = 0
                for j in range(5):
                    if max_n < label_n[j]:
                        max_n = label_n[j]
                        label = j
                o.write(correct_items[0] + '\t' + str(label) + '\t' + correct_items[1] + '\n')
                phrase_items = []
                correct_items = []


if __name__ == '__main__':
    main('out_with_predict.tsv', 'majority2.tsv')

# -*- coding: utf-8 -*-


def main():
    correct_n = 0.0
    wrong_n = 0.0
    with open('majority2.tsv') as f:
        for line in f:
            line = line.replace('\n', '')
            items = line.split('\t')
            if items[0] == items[1]:
                correct_n += 1
            else:
                wrong_n += 1

    print(correct_n / (correct_n + wrong_n))

if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-


def sigmoid5_consult_softmax(sentence_result):
    """
    sigmoid5の出力について、softmaxによってまとめる
    :param sentence_result: 文に対する出力の集合
    :return: 予測極性
    """
    # 全てのフレーズを見て、そのうち最も確信度の大きいラベルを文の予測ラベルとする
    max_sentence_conviction_rate = 0.0
    sentence_label = 0

    # フレーズ毎に参照していく
    for window_result in sentence_result:
        for phrase_result in window_result:
            phrase_conviction_rate = 0.0  # フレーズにおける確信度
            phrase_label = 0  # フレーズの予測ラベル

            # フレーズ内のラベル予測確率の最大値とそのラベルを求める
            for i in range(4):
                phrase_conviction_rate = abs(phrase_result[i] - 0.5)  # 出力と閾値の乖離具合を計算
                if phrase_result[i] >= 0.5:  # 0.5の閾値を超えれば、次の出力を見に行く
                    phrase_label = i+1
                else:
                    break  # 閾値を超えなければ、予測ラベルを確定

            # 今参照しているフレーズと、これまでのフレーズについての結果と比較し、より確率の高いラベルを文の予測ラベルとする
            if max_sentence_conviction_rate < phrase_conviction_rate:
                max_sentence_conviction_rate = phrase_conviction_rate
                sentence_label = phrase_label

    return sentence_label, max_sentence_conviction_rate


def main(input_file, output_file):
    with open(input_file, 'r') as i, open(output_file, 'w') as o:
        for line in i:
            line = line.replace('\n', '')
            items = line.split('\t')

            if len(items) != 5:
                o.write(line + '\n')
            else:
                result = sigmoid5_consult_softmax([[[float(items[0]), float(items[1]), float(items[2]), float(items[3])]]])
                shaped_items = [str(round(float(pred), 10)).ljust(13, ' ') for pred in items[:4]]
                o.write(str(result[0]) + '\t')
                o.write(str(round(result[1], 7)).ljust(10, ' ') + '\t')
                o.write('\t'.join(shaped_items) + '\t' + items[4] + '\n')

if __name__ == '__main__':
    main('out.tsv', 'out_with_predict.tsv')

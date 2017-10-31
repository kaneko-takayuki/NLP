# -*- coding: utf-8 -*-


def judge_evaluation(data):
    """
    与えられた確率のうち、最も高い確率値とそのラベルを返す
    :param data: 確率データ
    :return: (予測ラベル, 確率)
    """
    max_predict = 0.0  # 最も高い確率値
    predict_label = 0  # 予測ラベル

    for i in range(len(data)):
        if max_predict < data[i]:
            max_predict = data[i]
            predict_label = i

    return predict_label, max_predict


def ffnn_consult_softmax(sentence_result):
    """
    softmaxによってまとめる
    :param sentence_result: 文に対する出力の集合
    :return: 予測極性
    """
    max_abs = 0.0
    predict_label = 0

    # フレーズ毎に参照していく
    for window_result in sentence_result:
        for phrase_result in window_result:
            # 見てるフレーズが、ネガティブの確率の方が高く、その確率値が暫定値より高い時
            if (phrase_result[0] > phrase_result[1]) and (max_abs < phrase_result[0]):
                max_abs = phrase_result[0]
                predict_label = 0
            # 参照しているフレーズがポジティブの確率の方が高く、その確率値が暫定値より高い時
            if (phrase_result[0] < phrase_result[1]) and (max_abs < phrase_result[1]):
                max_abs = phrase_result[1]
                predict_label = 1

    return predict_label


def ffnn_consult_majority(sentence_result):
    """
    多数決によってまとめる
    :param sentence_result: 文に対する出力の集合
    :return: 予測極性
    """
    positive_n = 0  # ポジティブ判定の数
    negative_n = 0  # ネガティブ判定の数

    # フレーズ毎に参照していく
    for window_result in sentence_result:
        for phrase_result in window_result:
            if phrase_result[0] > phrase_result[1]:
                negative_n += 1
            else:
                positive_n += 1

    if positive_n >= negative_n:
        return 1  # ポジティブ判定
    else:
        return 0  # ネガティブ判定


def ffnn_consult_majority5(sentence_result):
    """
    多数決によってまとめる(5段階評価)
    :param sentence_result: 文に対する出力の集合
    :return: 予測極性
    """
    # 評価の個数をそれぞれ数える
    label_n = [0 for _ in range(5)]
    predict_label = 0  # 予測ラベル
    max_n = 0  # 予測フレーズの個数

    # フレーズ毎に参照していく
    for window_result in sentence_result:
        for phrase_result in window_result:
            predict_label, max_predict = judge_evaluation(phrase_result[0:5])
            label_n[predict_label] += 1

    for i in range(5):
        if max_n < label_n[i]:
            max_n = label_n[i]
            predict_label = i

    return predict_label


def svm_consult_softmax(sentence_result):
    """
    softmaxによってまとめる
    :param sentence_result: 文に対する出力の集合
    :return: 予測極性
    """
    max_abs = 0.0
    predict_label = 0

    # フレーズ毎に参照していく
    for window_result in sentence_result:
        for phrase_result in window_result:
            # 見てるフレーズが、ネガティブの確率の方が高く、その確率値が高い時
            if (phrase_result[0] == 0) and (max_abs < abs(phrase_result[1])):
                max_abs = abs(phrase_result[1])
                predict_label = 0
            # 参照しているフレーズがポジティブの確率の方が高く、その確率値が高い時
            if (phrase_result[0] == 1) and (max_abs < abs(phrase_result[1])):
                max_abs = abs(phrase_result[1])
                predict_label = 1

    return predict_label


def svm_consult_majority(sentence_result):
    """
    多数決によってまとめる
    :param sentence_result: 文に対する出力の集合
    :return: 予測極性
    """
    positive_n = 0  # ポジティブ判定の数
    negative_n = 0  # ネガティブ判定の数

    # フレーズ毎に参照していく
    for window_result in sentence_result:
        for phrase_result in window_result:
            if phrase_result[0] == 0:
                negative_n += 1
            else:
                positive_n += 1

    if positive_n >= negative_n:
        return 1  # ポジティブ判定
    else:
        return 0  # ネガティブ判定


def sigmoid5_consult_majority(sentence_result):
    """
    sigmoid5の出力について、多数決によってまとめる
    :param sentence_result: 文に対する出力の集合
    :return: 予測極性
    """
    # 1〜5の予測ラベルの個数
    label_n = [0 for _ in range(5)]

    # フレーズ毎に参照していく
    for window_result in sentence_result:
        for phrase_result in window_result:
            max_phrase_label_pred = 0.0  # フレーズにおけるラベル確率の最大値
            phrase_label = 0  # フレーズの予測ラベル
            for i in range(5):
                if max_phrase_label_pred < phrase_result[i]:
                    max_phrase_label_pred = phrase_result[i]
                    phrase_label = i
            label_n[phrase_label] += 1

    # 1〜5の予測ラベルの個数を比較し、文章の予測ラベルを返す
    max_sentence_label_n = 0.0
    sentence_label = 0
    for i in range(5):
        if max_sentence_label_n < label_n[i]:
            sentence_label = i

    return sentence_label


def sigmoid5_consult_softmax(sentence_result):
    """
    sigmoid5の出力について、softmaxによってまとめる
    :param sentence_result: 文に対する出力の集合
    :return: 予測極性
    """
    # 全てのフレーズを見て、そのうち最も確率の高いラベルを文の予測ラベルとする
    max_sentence_label_pred = 0.0
    sentence_label = 0

    # フレーズ毎に参照していく
    for window_result in sentence_result:
        for phrase_result in window_result:
            max_phrase_label_pred = 0.0  # フレーズにおけるラベル確率の最大値
            phrase_label = 0  # フレーズの予測ラベル

            # フレーズ内のラベル予測確率の最大値とそのラベルを求める
            for i in range(5):
                if max_phrase_label_pred < phrase_result[i]:
                    max_phrase_label_pred = phrase_result[i]
                    phrase_label = i

            # 今参照しているフレーズと、これまでのフレーズについての結果と比較し、より確率の高いラベルを文の予測ラベルとする
            if max_sentence_label_pred < max_phrase_label_pred:
                max_sentence_label_pred = max_phrase_label_pred
                sentence_label = phrase_label

    return sentence_label

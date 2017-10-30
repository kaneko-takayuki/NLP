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
    label_n = [0 for _ in 5]
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

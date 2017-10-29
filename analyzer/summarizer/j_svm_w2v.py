# -*- coding: utf-8 -*-

from analyzer.summarizer import consult_function as cf


def summarize_svm_w2v(input_files, output_file):
    """
    SVMで出した出力をまとめる
    :param input_files: まとめる対象のファイルリスト
    :param output_file: 出力ファイル名
    :return: なし
    """
    # 出力ファイルの初期化
    with open(output_file, 'w'):
        pass

    # 文章数の計算
    sentence_n = 0
    with open(input_files[0], 'r') as f:
        for line in f:
            if len(line.strip()) == 0:
                sentence_n += 1

    # ファイルポインタをそれぞれ準備
    file_pointer = []
    for input_file in input_files:
        file_pointer.append(open(input_file))

    # 文章毎に結果をまとめる
    for _ in range(sentence_n):
        # 結果をまとめるリスト
        sentence_result = []
        for pointer in file_pointer:
            window_result = []
            for line in pointer:
                line = line.strip()
                items = line.split('\t')

                # 空行の終わりは1文分のデータを読み終わったことを意味する
                if len(line) == 0:
                    break

                # 文章の最初の行
                if len(items) == 2:
                    correct_items = items
                    continue

                window_result.append([float(items[0]), float(items[1]), items[2]])
            sentence_result.append(window_result)

        # それぞれの結果から一つに束ねる
        predict_label = cf.consult_func(sentence_result=sentence_result)

        # 結果ファイルに出力
        with open(output_file, "a") as o:
            o.write(str(correct_items[0]) + "\t" + str(predict_label) + "\t" + correct_items[1] + "\n")

    # ファイルポインタを全てクローズ
    for pointer in file_pointer:
        pointer.close()

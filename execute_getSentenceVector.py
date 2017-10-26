# -*- coding: utf-8 -*-

import LSTMRunner
import Constant

if __name__ == "__main__":
    input_path = Constant.BASE_PATH + "tsukuba_corpus/data/"
    output_path = Constant.BASE_PATH + "tsukuba_corpus/LSTM_SentenceVector_data/"

    runner = LSTMRunner.LSTMRunner()
    runner.init_model(n_in=300, n_units=300, n_out=2, gpu=0)
    runner.load(file_name="tsukuba_corpus/LSTM_W2V/model/cross_validation1/epoch50_model.npz")

    for k in range(1, 6):
        input_file = input_path + "dataset" + str(k) + ".tsv"
        output_file = output_path + "dataset" + str(k) + ".tsv"
        with open(output_file, "w"):
            pass

        with open(input_file, "r") as f:
            for line in f:
                items = line.split("\t")
                items[5] = items[5].strip()
                print(items[5])
                compression_vector = runner.get_compression_vector(items[5])[0]
                compression_vector = [str(v) for v in compression_vector]
                print("ベクトル次元数: {}".format(len(compression_vector)))
                with open(output_file, "a") as o:
                    o.write("\t".join(items) + "\t" + " ".join(compression_vector) + "\n")

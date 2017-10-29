#!/usr/bin/env bash

file1=tsukuba_corpus/ffnn_w2v/multi_window357/zero/summarized_out/majority/cross_validation1/accuracy_file2.txt
file2=tsukuba_corpus/ffnn_w2v/multi_window357/zero/summarized_out/softmax/cross_validation1/accuracy_file2.txt
file3=tsukuba_corpus/ffnn_w2v/multi_window357/random/summarized_out/softmax/cross_validation1/accuracy_file2.txt

python exec/analyze/exec_draw_graph_j_ffnn_w2v.py -f ${file1} ${file2} -x 100

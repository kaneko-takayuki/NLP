#!/usr/bin/env bash

#file1=tsukuba_corpus/ffnn_w2v/multi_window357/zero/summarized_out/majority/cross_validation1/accuracy_file2.txt
#file2=tsukuba_corpus/ffnn_w2v/multi_window357/zero/summarized_out/softmax/cross_validation1/accuracy_file2.txt
#file3=tsukuba_corpus/ffnn_w2v/multi_window357/random/summarized_out/softmax/cross_validation1/accuracy_file2.txt

file1=amazon_corpus/FFNN_W2V/Window3/random/summarized_out/majority/cross_validation1/accuracy_file.txt
file2=amazon_corpus/FFNN_W2V/Window5/random/summarized_out/majority/cross_validation1/accuracy_file.txt
file3=amazon_corpus/FFNN_W2V/Window7/random/summarized_out/majority/cross_validation1/accuracy_file.txt
file4=amazon_corpus/FFNN_W2V/Window9/random/summarized_out/majority/cross_validation1/accuracy_file.txt
file5=amazon_corpus/FFNN_W2V/Window11/random/summarized_out/majority/cross_validation1/accuracy_file.txt
file6=amazon_corpus/FFNN_W2V/Window13/random/summarized_out/majority/cross_validation1/accuracy_file.txt

label1=window3
label2=window5
label3=window7
label4=window9
label5=window11
label6=window13

python exec/analyze/exec_draw_graph_j_ffnn_w2v.py -f ${file1} ${file2} ${file3} ${file4} ${file5} ${file6} -l ${label1} ${label2} ${label3} ${label4} ${label5} ${label6} -x 100
python exec/analyze/exec_draw_graph_j_ffnn_w2v.py -f ${file1} ${file2} ${file3} ${file4} ${file5} ${file6} -l ${label1} ${label2} ${label3} ${label4} ${label5} ${label6} -x 100

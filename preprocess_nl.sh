#!/bin/bash

DATA_DIR="/data/p289796/nl-data"
# preprocess
python preprocess_noedge_sbn.py $DATA_DIR/train.txt.graph.normal
python preprocess_noedge_sbn.py $DATA_DIR/dev.txt.graph.normal
python preprocess_noedge_sbn.py $DATA_DIR/test.txt.graph.normal
/data/p289796/mosesdecoder/scripts/tokenizer/tokenizer.perl -l nl < ${DATA_DIR}/train.txt.raw > ${DATA_DIR}/train.txt.raw.tok -no-escape 
/data/p289796/mosesdecoder/scripts/tokenizer/tokenizer.perl -l nl < ${DATA_DIR}/dev.txt.raw > ${DATA_DIR}/dev.txt.raw.tok -no-escape 
/data/p289796/mosesdecoder/scripts/tokenizer/tokenizer.perl -l nl < ${DATA_DIR}/test.txt.raw > ${DATA_DIR}/test.txt.raw.tok -no-escape 

python preprocess.py -train_src $DATA_DIR/train.txt-src-nodes.txt \
            -train_node1  $DATA_DIR/train.txt-src-node1.txt\
            -train_node2  $DATA_DIR/train.txt-src-node2.txt\
            -train_tgt    $DATA_DIR/train.txt.raw.tok\
            -valid_src   $DATA_DIR/dev.txt-src-nodes.txt\
            -valid_node1 $DATA_DIR/dev.txt-src-node1.txt\
            -valid_node2 $DATA_DIR/dev.txt-src-node2.txt\
            -valid_tgt   $DATA_DIR/dev.txt.raw.tok\
            -save_data   $DATA_DIR/gcn_exp \
            -data_type graph\
            -src_words_min_frequency 1 \
            -tgt_words_min_frequency 1 \
            -src_seq_length 1000 \
            -tgt_seq_length 1000 \
            -dynamic_dict 


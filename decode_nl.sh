#!/bin/bash

if [ "$#" -lt 6 ]; then
  echo "./decode.sh <gpuid> <model> <nodes-file> <node1-file> <node2-file> <output>"
  exit 2
fi


DATA_DIR="/data/p289796/nl-data"
DATASET=test

GPUID=$1
MODEL=$2
NODES_FILE=$3
NODE1_FILE=$4
NODE2_FILE=$5
OUTPUT=$6

export CUDA_VISIBLE_DEVICES=${GPUID}
export OMP_NUM_THREADS=10

python -u translate.py -model ${MODEL}\
  -data_type graph  \
  -src ${NODES_FILE} \
  -src_node1 ${NODE1_FILE} \
  -src_node2 ${NODE2_FILE} \
  -output ${OUTPUT}\
  -replace_unk \
  -verbose \
  -gpu $GPUID \
  -batch_size 50 \
  -max_length 300 \
  -dynamic_dict \
  -beam_size 5 \
  -block_ngram_repeat 3


/data/p289796/mosesdecoder/scripts/tokenizer/detokenizer.perl -l en < ${OUTPUT} > ${OUTPUT}.detok.txt
#python data_preprocess/postprocess_deAnony.py ${OUTPUT}.detok.txt ${DATA_DIR}/${DATASET}.txt.alignment ${OUTPUT}.detok.txt.dealign
python ~/project/e2e-metrics/measure_scores.py ${DATA_DIR}/${DATASET}.txt.raw ${OUTPUT}.detok.txt


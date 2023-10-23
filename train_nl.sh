#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=10:00:00
#SBATCH --mem=100GB

if [ "$#" -lt 5 ]; then
  echo "./train.sh <gpuid> <gnn_type> <gnn_layers> <start_decay_steps> <decay_steps>"
  exit 2
fi



GPUID=$1
GNNTYPE=$2
GNNLAYERS=$3
STARTDECAYSTEPS=$4
DECAYSTEPS=$5
STEPS=1000
EPOCHS=$((${STEPS}*15))
DATA_DIR="/data/p289796/nl-data"

mkdir -p ${DATA_DIR}/models
#export CUDA_VISIBLE_DEVICES=${GPUID}
export OMP_NUM_THREADS=10

#echo ${DATA_DIR}/models/${DATASET}-${GNNTYPE}-2

python -u train.py -data ${DATA_DIR}/gcn_exp \
-save_model ${DATA_DIR}/models/${GNNTYPE}-2 \
-rnn_size 900 -word_vec_size 300 -train_steps ${EPOCHS} -optim adam \
-valid_steps ${STEPS} \
-valid_batch_size 1 \
-encoder_type graph \
-gnn_type ${GNNTYPE} \
-gnn_layers ${GNNLAYERS} \
-decoder_type rnn \
-learning_rate 0.001 \
-dropout 0.5 \
-copy_attn -copy_attn_type general -batch_size 20 \
-save_checkpoint_steps ${STEPS} \
-start_decay_steps ${STARTDECAYSTEPS} \
-decay_steps ${DECAYSTEPS} \
-layers 2 \
-global_attention general \
-gpu_ranks 0 \
-keep_checkpoint 3

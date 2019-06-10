#!/bin/bash

# from experiment file
# Mandatory SRVR_ADDR, SRVR_DEST_DIR
# Mandatory BATCH_SIZE, SEQ_LENGTH
# Optional EXP_NAME, GPUS_NO, GPUS_MODEL, CHECKPOINT
[ -f experiment ] && source experiment

BATCH_SIZE=${BATCH_SIZE:-12}
SEQ_LENGTH=${SEQ_LENGTH:-384}

BERT_DIR=${BERT_DIR:-`pwd`/models/uncased}
SQUAD_DIR=${SQUAD_DIR:-`pwd`/squad}
OUTPUT_DIR=${OUTPUT_DIR:-`pwd`/squad_output}
CHECKPOINT=${CHECKPOINT}

TRAIN=${TRAIN:-True}
THRESH=${THRESH:-'0'}

TRAIN_FILE=${TRAIN_FILE:-$SQUAD_DIR/train-v2.0.json}
PREDICT_FILE=${PREDICT_FILE:-$SQUAD_DIR/dev-v2.0.json}

TRAIN_EPOCHS=${TRAIN_EPOCHS:-2.0}

cd bert

python run_squad.py \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$CHECKPOINT \
  --do_train=$TRAIN \
  --train_file=$TRAIN_FILE \
  --do_predict=True \
  --predict_file=$PREDICT_FILE \
  --train_batch_size=$BATCH_SIZE \
  --learning_rate=3e-5 \
  --num_train_epochs=$TRAIN_EPOCHS \
  --max_seq_length=$SEQ_LENGTH \
  --doc_stride=128 \
  --output_dir=$OUTPUT_DIR \
  --use_tpu=False \
  --version_2_with_negative=True \
  --null_score_diff_threshold=$THRESH
cd -



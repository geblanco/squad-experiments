#!/bin/bash

# from experiment file
# Mandatory SRVR_ADDR, SRVR_DEST_DIR
# Mandatory BATCH_SIZE, SEQ_LENGTH
# Optional EXP_NAME, GPUS_NO, GPUS_MODEL
[ -f experiment ] && source  experiment
[[ -z SRVR_ADDR || -z SRVR_DEST_DIR ]] && (echo 'No backup space'; exit 1)
[[ -z BATCH_SIZE || -z SEQ_LENGTH ]] && (echo 'Need experiment variables'; exit 2)

BERT_DIR=`pwd`/models/uncased
SQUAD_DIR=`pwd`/squad
OUTPUT_DIR=`pwd`/squad_output

cd bert
python run_squad.py \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$BERT_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v2.0.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v2.0.json \
  --train_batch_size=$BATCH_SIZE \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=$SEQ_LENGTH \
  --doc_stride=128 \
  --output_dir=$OUTPUT_DIR \
  --use_tpu=False \
  --version_2_with_negative=True \
cd -



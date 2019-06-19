#!/bin/bash

# from experiment file
#   BERT_DIR
#   CHECKPOINT
#   TRAIN
#   TRAIN_FILE
#   PREDICT_FILE
#   BATCH_SIZE
#   TRAIN_EPOCHS
#   SEQ_LENGTH
#   OUTPUT_DIR
#   THRESH

exp=$1
[ -f $exp ] && source $exp

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

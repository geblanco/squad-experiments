#!/bin/bash

# from experiment file
#   TRAIN_FILE
#   VALIDATION_FILE
#   PREDICT_FILE
#   TRAIN
#   BATCH_SIZE
#   LEARNING_RATE
#   TRAIN_EPOCHS
#   OUTPUT_DIR
#   SEQ_LENGTH
#   LSTM_HIDDEN_SIZE
#   EMBEDDINGS_FILE
#   EMBEDDINGS_SI

exp=$1
[ ! -f $exp ] && exit 1;
source $exp

cd src

pip install --upgrade pip
pip install -r requirements.txt

# --max_words
# --max_query_length
python voting_keras.py \
  --train_file=$TRAIN_FILE \
  --validate_file=$VALIDATE_FILE \
  --predict_file=$PREDICT_FILE \
  --do_train=$TRAIN \
  --do_predict=True \
  --batch_size=$BATCH_SIZE \
  --learning_rate=$LEARNING_RATE \
  --num_train_epochs=$TRAIN_EPOCHS \
  --output_dir=$OUTPUT_DIR \
  --max_seq_length=$SEQ_LENGTH \
  --lstm_hidden_size=$LSTM_HIDDEN_SIZE \
  --embeddings_file=$EMBEDDINGS_FILE \
  --embeddings_size=$EMBEDDINGS_SIZE


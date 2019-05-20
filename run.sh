#!/bin/bash

# from experiment file
# Mandatory SRVR_ADDR, SRVR_DEST_DIR
# Mandatory BATCH_SIZE, SEQ_LENGTH
# Optional EXP_NAME, GPUS_NO, GPUS_MODEL
[ -f experiment ] && source  experiment
[[ -z SRVR_ADDR || -z SRVR_DEST_DIR ]] && (echo 'No backup space'; exit 1)

OUTPUT_DIR=`pwd`/squad_output
TRAIN=${1:-True}
THRESH=${2:-'0'}
BACKUP=${3:-0}
POWEROFF=${4:-0}

if [[ -z $DOCKERIZE ]]; then
  time -o $OUTPUT_DIR/run_time.txt ./run_squad.sh $TRAIN $THRESH
else
  time -o $OUTPUT_DIR/run_time.txt (nvidia-docker run \
    -v `pwd`:/workspace \
    nvcr.io/nvidia/tensorflow:19.02-py3 \
    /workspace/run_squad.sh $TRAIN $THRESH \
    2>&1) \
  | tee $OUTPUT_DIR/train_log
fi

# backup data
[ $BACKUP -eq 0 ] || rsync -avrzP $OUTPUT_DIR $SRVR_ADDR:$SRVR_DEST_DIR
[ $POWEROFF -eq 0 ] || sudo poweroff


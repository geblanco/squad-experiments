#!/bin/bash

# from experiment file
# Mandatory SRVR_ADDR, SRVR_DEST_DIR
# Mandatory BATCH_SIZE, SEQ_LENGTH
# Optional EXP_NAME, GPUS_NO, GPUS_MODEL
[ -f experiment ] && source  experiment
[[ -z SRVR_ADDR || -z SRVR_DEST_DIR ]] && (echo 'No backup space'; exit 1)

OUTPUT_DIR=`pwd`/squad_output

time nvidia-docker run \
  -v `pwd`:/workspace \
  nvcr.io/nvidia/tensorflow:19.02-py3 \
  /workspace/run_docker.sh \
| tee $OUTPUT_DIR/train_log

# --null_score_diff_threshold=$THRESH, where $THRESH is the output with do_train=True
cd -

# backup data
rsync -avrzP $OUTPUT_DIR $SRVR_ADDR:$SRVR_DEST_DIR
sudo poweroff


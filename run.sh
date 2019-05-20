#!/bin/bash

POWEROFF=${1:-0}

[ -f experiment ] && source  experiment

if [[ ! -z $BACKUP && $BACKUP -eq 1 ]]; then
  [[ -z SRVR_ADDR || -z SRVR_DEST_DIR ]] && (echo 'No backup space'; exit 1)
fi

OUTPUT_DIR=${OUTPUT_DIR:-`pwd`/squad_output}
TRAIN=${TRAIN:-True}
THRESH=${THRESH:-'0'}

if [[ -z $DOCKERIZE ]]; then
  time -o $OUTPUT_DIR/run_time.txt ./run_squad.sh
else
  time -o $OUTPUT_DIR/run_time.txt (nvidia-docker run \
    -v `pwd`:/workspace \
    nvcr.io/nvidia/tensorflow:19.02-py3 \
    /workspace/run_squad.sh \
    2>&1) \
  | tee $OUTPUT_DIR/train_log
fi

# backup data
[ $BACKUP -eq 0 ] || rsync -avrzP $OUTPUT_DIR $SRVR_ADDR:$SRVR_DEST_DIR
[ $POWEROFF -eq 0 ] || sudo poweroff


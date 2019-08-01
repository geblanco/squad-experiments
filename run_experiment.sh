#!/bin/bash

exp=$1
log_prefix=${2:-train}

[ -f $exp ] && source $exp

if [[ -z $BACKUP && $BACKUP -eq 1 ]]; then
  [[ -z SRVR_ADDR || -z SRVR_DEST_DIR ]] && (echo 'No backup space'; exit 1)
fi

if [[ -z $OUTPUT_DIR ]]; then
  echo "No output directory!"
  exit 2
fi

POWEROFF=${POWEROFF:-0}
BACKUP=${BACKUP:-0}

if [[ -z $DOCKERIZE || $DOCKERIZE -eq 0 ]]; then
  { time ./run_squad.sh $exp 2>&1 \
    | tee "${OUTPUT_DIR}/${log_prefix}.log";
  } 2>"${OUTPUT_DIR}/${log_prefix}.run_time"
else
  { time nvidia-docker run \
    -v `pwd`:/workspace \
    nvcr.io/nvidia/tensorflow:19.02-py3 \
    /workspace/run_squad.sh $exp \
    2>&1 \
    | tee "${OUTPUT_DIR}/${log_prefix}.log"; \
  } 2>"${OUTPUT_DIR}/${log_prefix}.run_time"
fi

# backup data
[ $BACKUP -eq 0 ] || rsync -avrzP $OUTPUT_DIR $SRVR_ADDR:$SRVR_DEST_DIR
[ $POWEROFF -eq 0 ] || sudo poweroff


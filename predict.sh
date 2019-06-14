#!/bin/bash

# set -e

if [[ $# -gt 0 ]]; then
  predictions=$@
else
  predictions=$(ls -d predictions/*/ | cut -f2 -d'/')
fi

for prediction in ${predictions[@]}; do
  experiments=$(ls predictions/${prediction}/*.experiment)
  for exp in ${experiments[@]}; do
    output="${exp%.*}_data"
    if [[ -f "${output}/predictions.json" ]]; then
      continue
    fi

    if [[ -d $output ]]; then
      rm -rf $output
    fi
    
    mkdir $output

    source $exp
    BACKUP=${BACKUP:-0}
    if [[ -z $BACKUP && $BACKUP -eq 1 ]]; then
      [[ -z SRVR_ADDR || -z SRVR_DEST_DIR ]] && (echo 'No backup space'; exit 1)
    fi

    if [[ -z $DOCKERIZE || $DOCKERIZE -eq 0 ]]; then
      { time ./run_squad.sh $exp 2>&1 | tee $OUTPUT_DIR/prediction_log; } 2>$OUTPUT_DIR/prediction_time.txt
    else
      # in case of docker, just copy the experiment file
      cp $exp experiment
      { time nvidia-docker run \
        -v `pwd`:/workspace \
        nvcr.io/nvidia/tensorflow:19.02-py3 \
        /workspace/run_squad.sh \
        2>&1 \
        | tee $OUTPUT_DIR/prediction_log; \
      } 2>$OUTPUT_DIR/prediction_time.txt
    fi

    cp $OUTPUT_DIR/nbest_predictions.json "${output}/nbest_predictions.json"
    cp $OUTPUT_DIR/null_odds.json "${output}/null_odds.json"
    cp $OUTPUT_DIR/predictions.json "${output}/predictions.json"

    # backup data
    [ $BACKUP -eq 0 ] || rsync -avrzP $output $SRVR_ADDR:$SRVR_DEST_DIR
  done
done

#!/bin/bash


correct_thr=${1:-0}; shift

if [[ $# -gt 0 ]]; then
  predictions=$@
else
  predictions=$(ls -d predictions/*/ | cut -f2 -d'/')
fi

set -e

eval_script=`pwd`/squad/evaluate-v2.0.py
for prediction in ${predictions[@]}; do
  experiments=$(ls predictions/${prediction}/*.experiment)
  for exp in ${experiments[@]}; do
    prediction_data="${exp%.*}_data"
    if [[ -d $prediction_data ]]; then
      source $exp
      output=${prediction_data}/results.json
      # get new threshold
      echo "$(basename $PREDICT_FILE) $(basename ${prediction_data})/predictions.json --na-prob-file $(basename ${prediction_data})/null_odds.json"
      new_thrs=$(python $eval_script $PREDICT_FILE ${prediction_data}/predictions.json --na-prob-file ${prediction_data}/null_odds.json | tee $output | grep "best_f1_thresh" | awk '{print $2}')
      if [[ $correct_thr -ne 0 ]]; then
        # remove threshold from experiment file
        sed -e '/THRESH=/d' --in-place=_old "${exp}"
        echo "THRESH='$new_thrs'" >>  "${exp}"
      fi
    fi
  done
done

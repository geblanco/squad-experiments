#!/bin/bash


data_dir=${1:-results}; shift
overwrite=${1:-0}; shift
correct_thr=${1:-0}; shift
predictions=$(ls -d $data_dir/*/ | cut -f2 -d'/')

set -e

eval_script=`pwd`/squad/evaluate-v2.0.py
for prediction in ${predictions[@]}; do
  experiments=$(ls ${data_dir}/${prediction}/*.experiment)
  for exp in ${experiments[@]}; do
    prediction_data="${exp%.*}_out"
    if [[ -d $prediction_data ]]; then
      source $exp
      output=${prediction_data}/results.json
      exact_scores=${prediction_data}/exact_scores.json
      f1_scores=${prediction_data}/f1_scores.json
      # get new threshold
      if [[ -f $output && $overwrite -eq 0 ]]; then 
        echo "Skipping $(basename ${prediction_data}), use overwrite=1 to force it anyway."
        continue
      fi
      echo "$(basename $PREDICT_FILE) $(basename ${prediction_data})/predictions.json --na-prob-file $(basename ${prediction_data})/null_odds.json"
      python_args="$eval_script $PREDICT_FILE ${prediction_data}/predictions.json"
      python_args+=" --na-prob-file ${prediction_data}/null_odds.json"
      python_args+=" --exact_question_scores_file $exact_scores"
      python_args+=" --f1_question_scores_file $f1_scores"
      new_thrs=$(python ${python_args} | tee $output | grep "best_f1_thresh" | awk '{print $2}')
      if [[ $correct_thr -ne 0 ]]; then
        # remove threshold from experiment file
        sed -e '/THRESH=/d' --in-place=_old "${exp}"
        echo "THRESH='$new_thrs'" >>  "${exp}"
      fi
    fi
  done
done

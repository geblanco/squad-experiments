#!/bin/bash

set -e

if [[ $# -lt 1 ]]; then
  echo "Usage ./eval_experiments.sh <filelist> [<apply_threshold>]"
  exit 0
fi

# filelist must point to the results folder directly, without trailing slash
# example ./results/newsqa/newsqa_pred_base_model_out
# parse avoiding comments
experiments=$(grep "#--" -v $1)
apply_threshold=${2:-0}

eval_script=`pwd`/squad/evaluate-v2.0.py
for exp in ${experiments[@]}; do
  f_name=$(basename ${exp%_out})
  f_path=$(basename `dirname $exp`)/$f_name
  experiment="experiments/${f_path}.experiment"
  data="${exp}"
  if [[ ! -f $experiment ]]; then
    echo "Unable to find $experiment"
    continue
  fi
  source $experiment
  if [[ ! -f $PREDICT_FILE ]]; then
    echo "Unable to find $PREDICT_FILE"
    continue
  fi
  output=${data}/results_no_thresh.json
  exact_scores=${data}/exact_scores.json
  f1_scores=${data}/f1_scores.json
  echo "$(basename $PREDICT_FILE) $(basename ${data})/predictions.json --na-prob-file $(basename ${data})/null_odds.json"
  # Calculate threshold
  python_args="$eval_script $PREDICT_FILE ${data}/predictions.json"
  python_args+=" --na-prob-file ${data}/null_odds.json"
  python_args+=" --exact_question_scores_file $exact_scores"
  python_args+=" --f1_question_scores_file $f1_scores"
  thresh=$(python ${python_args} | tee $output | grep "best_f1_thresh" | awk '{print $2}')
  if [[ "$apply_threshold" -eq 0 ]]; then
    mv $output ${data}/results.json
  else
    # get f1 result
    result=$(grep '"exact": ' $output | awk '{print $NF}' | tr -d ',')
    new_output=${data}/result_with_thresh.json
    python_args+=" --na-prob-thresh $thresh"
    # calculate f1 result with new threshold
    python ${python_args} > $new_output
    new_result=$(grep '"exact": ' $new_output | awk '{print $NF}' | tr -d ',')
    to_rm=$new_output
    to_mv=$output
    if [[ $(echo "$new_result > $result" | bc) -eq 1 ]]; then
      to_rm=$output
      to_mv=$new_output
    fi
    rm $to_rm
    mv $to_mv ${data}/results.json
  fi
done

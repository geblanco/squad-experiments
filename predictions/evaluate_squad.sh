#!/bin/bash

preds_dir=`pwd`
root_dir=$(dirname $preds_dir)
eval_script=${root_dir}/squad/evaluate-v2.0.py
predict_files=$(ls *.experiment)

cd $root_dir
for prediction in ${predict_files[@]}; do
  source "${preds_dir}/${prediction}"
  prediction_data="${preds_dir}/${prediction%.*}_data"
  output=${prediction_data}/results.json
  # get new threshold
  new_thrs=$(python $eval_script $PREDICT_FILE ${prediction_data}/predictions.json --na-prob-file ${prediction_data}/null_odds.json | tee $output | grep "best_f1_thresh" | awk '{print $2}')
  # remove threshold from experiment file
  sed -e '/THRESH=/d' --in-place=_old "${preds_dir}/${prediction}"
  echo "THRESH='$new_thrs'" >>  "${preds_dir}/${prediction}"
done

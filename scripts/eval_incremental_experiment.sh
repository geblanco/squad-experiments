#!/bin/bash

if [[ "$#" -lt 2 ]]; then
  echo "Usage: evaluate_incremental_experiment.sh <results folder> <dataset> [, <dataset>...]"
  exit 0
fi

dir=$1; shift
dataset="/tmp/merged_dataset_${RANDOM}.json"
echo "Merging datasets to $dataset"
python ./scripts/merge_datasets.py -d $@ -o $dataset

experiments=($(ls -d $dir/predictions_*))
null_odds=($(ls -d $dir/null_odds_*))
if [[ "${#experiments[@]}" -ne "${#null_odds[@]}" ]]; then
  echo "Number of prediction and null odd files differ!"
  exit 1
fi

nof_experiments=${#experiments[@]}
for ((idx=0; idx<$nof_experiments; idx++)); do
  exp=${experiments[$idx]}
  null_odd_file=${null_odds[$idx]}
  exp_no=$(( $idx +1 ))
  echo "Exp: $exp"
  echo "Null odds: $null_odd_file"
  id_list="$dir/id_list_${exp_no}.json"
  exp_dataset="$dir/dataset_${exp_no}.json"
  results_file="$dir/results_${exp_no}.json"
  aux_results_file="$dir/results_${exp_no}_aux.json"
  echo " -> Compose id list from sample $exp_no"
  python ./scripts/predictions_to_id_list.py -p $exp -o $id_list
  echo " -> Compose dataset from sample $exp_no"
  python ./scripts/sample_dataset_from_ids.py -d $dataset -o $exp_dataset -i $id_list
  echo " -> Evaluate squad over sample $exp_no: $results_file"
  python squad/evaluate-v2.0.py $exp_dataset $exp --na-prob-file $null_odd_file > $results_file
  echo " -> Evaluate precision recall"
  python scripts/eval.py $exp_dataset $exp --merge $results_file > $aux_results_file
  mv $aux_results_file $results_file
done

#!/bin/bash

set -e 

for dir in $(cat filelist); do
  exp_file=${dir%_out}.experiment
  if [[ ! -f $exp_file ]]; then
    exp_family_name=$(dirname $exp_file)
    exp_family_name=$(basename $exp_family_name)
    exp_name=$(basename $exp_file)
    exp_dir_name=experiments/$exp_family_name/$exp_name
    echo $exp_family_name $exp_name $exp_dir_name
    if [[ -f $exp_dir_name ]]; then
      cp $exp_dir_name $exp_file
    else
      echo "Could not find $exp_file"
      break
    fi
  fi
  source $exp_file
  # echo "$PREDICT_FILE $file --merge $exp_dir/results.json"
  python scripts/eval.py $PREDICT_FILE $dir/predictions.json --merge $dir/results.json | python ./scripts/format_results.py --
done

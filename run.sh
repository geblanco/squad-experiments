#!/bin/bash

# run squad experiment
# ToDo := Should check user confirmation
#   heavy delete operations are carried out.
clean_data_dir() {
  local dir=$1
  if [[ -d $dir ]]; then
    echo "Delete $dir"
    rm -rf $dir
    echo "Create $dir"
    mkdir -p $dir
  fi
}

copy_model() {
  local checkpoint_file=$1
  local output=$2
  local model_chkpt=$(head -n 1 $checkpoint_file | grep -oE 'model.ckpt-[0-9]+')
  local model=$(dirname $checkpoint_file)/${model_chkpt}
  local model_data=${model}.data-00000-of-00001
  local model_index=${model}.index
  local model_meta=${model}.meta
  cp ${model_data} "${output}/model.ckpt.data-00000-of-00001"
  cp ${model_index} "${output}/model.ckpt.index"
  cp ${model_meta} "${output}/model.ckpt.meta"
}

# no errors accepted
set -e

experiments=($@)
for exp in ${experiments[@]}; do
  data_dir="${exp%.*}_out"
  clean_data_dir $data_dir
  source $exp
  if [[ ! -z $REUSE_MODEL ]]; then
    # copy model to avoid overriding the original squad one
    copy_model $REUSE_MODEL $OUTPUT_DIR
  fi
  # run experiment
  echo "Run $exp"
  ./run_experiment.sh $exp
done

# run all predictions after training all models
./predict.sh
./evaluate_squad.sh 1
./predict.sh

#!/bin/bash

# run squad experiment
# ToDo := Should check user confirmation
#   heavy delete operations are carried out.
clean_data_dir() {
  local dir=$1
  if [[ -d $dir ]]; then
    echo "Delete $dir"
    rm -rf $dir
  fi
  echo "Create $dir"
  mkdir -p $dir
}

copy_model() {
  local checkpoint_file=$1
  local output=$2
  local name=${3:-model}
  local model_chkpt=$(head -n 1 $checkpoint_file | grep -oE 'model.ckpt-[0-9]+')
  local model=$(dirname $checkpoint_file)/${model_chkpt}
  local model_data=${model}.data-00000-of-00001
  local model_index=${model}.index
  local model_meta=${model}.meta
  cp ${model_data} "${output}/${name}.ckpt.data-00000-of-00001"
  cp ${model_index} "${output}/${name}.ckpt.index"
  cp ${model_meta} "${output}/${name}.ckpt.meta"
}

# no errors accepted
set -e

experiments=($@)
for exp in ${experiments[@]}; do
  source $exp
  clean_data_dir $OUTPUT_DIR
  # run experiment
  echo "Run $exp"
  ./run_experiment.sh $exp
  if [[ $? -ne 0 ]]; then
    exit $?
  fi
  if [[ "$TRAIN" == "True" && ! -z $DROP_MODEL ]]; then
    drop_base=$(dirname $DROP_MODEL)
    drop_name=$(basename $DROP_MODEL)
    mkdir -p $drop_base
    echo "Copy model $OUTPUT_DIR/checkpoint $drop_base $drop_name"
    copy_model $OUTPUT_DIR/checkpoint $drop_base $drop_name
    unset DROP_MODEL
  fi
done

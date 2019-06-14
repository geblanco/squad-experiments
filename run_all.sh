#!/bin/bash

# run squad experiment
# ToDo := Should check user confirmation
#   heavy delete operations are carried out.

root_dir=`pwd`
experiments_dir=${root_dir}/experiments

squad_model_output=${root_dir}/squad_output

newsqa=${experiments_dir}/newsqa/newsqa_trained
triviaqa=${experiments_dir}/triviaqa/triviaqa_trained
mixed=${experiments_dir}/mixed/newsqa_and_triviaqa_trained

newsqa_model_output=${newsqa}_out
triviaqa_model_output=${triviaqa}_out
mixed_model_output=${mixed}_out

data_dirs=($newsqa_model_output $triviaqa_model_output $mixed_model_output)

# setup directories, no old models allowed
for data_dir in ${data_dirs[@]}; do
  if [[ -d $data_dir ]]; then
    echo "Delete $data_dir"
    rm -rf $data_dir
    echo "Create $data_dir"
    mkdir -p $data_dir
  fi
done

copy_model() {
  local checkpoint_file=$1
  local output=$2
  local model_chkpt=$(head -n 1 $checkpoint_file | grep -oE 'model.ckpt-[0-9]+')
  local model=$squad_model_output/$model_chkpt
  local model_data=${model}.data-00000-of-00001
  local model_index=${model}.index
  local model_meta=${model}.meta
  cp $model_data "${output}/model.ckpt.data-00000-of-00001"
  cp $model_index "${output}/model.ckpt.index"
  cp $model_meta "${output}/model.ckpt.meta"
}

# no errors accepted
set -e
echo "Run squad.experiment"
./run.sh squad.experiment

# squad model data
squad_checkpoint_file=$squad_model_output/checkpoint

# copy model to avoid overriding the original squad one
experiments=($newsqa $triviaqa)
for experiment in ${experiments[@]}; do
  echo "Copy model from $squad_model_output to ${experiment}_out"
  copy_model $squad_checkpoint_file ${experiment}_out
  # run experiment
  echo "Run ${experiment}.experiment"
  ./run.sh "${experiment}.experiment"
done

# mixed trains triviaqa over newsqa, copy newsqa 
newsqa_checkpoint_file=$newsqa_model_output/checkpoint

echo "Copy model from $newsqa_model_output to ${mixed}_out"
copy_model $newsqa_checkpoint_file ${mixed}_out

echo "Run ${mixed}.experiment"
./run.sh "${mixed}.experiment"

./predict.sh
./eval_squad.sh 1
./predict.sh

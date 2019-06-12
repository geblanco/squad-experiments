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

data_dirs=($squad_model_output $newsqa_model_output $triviaqa_model_output $mixed_model_output)

for data_dir in ${data_dirs[@]}; do
  if [[ -d $data_dir ]]; then
    echo "Delete $data_dir"
    rm -rf $data_dir
    echo "Create $data_dir"
    mkdir -p $data_dir
  fi
done

# no errors accepted
set -e
echo "Run squad.experiment"
./run.sh squad.experiment

# copy model to avoid overriding the original squad one
experiments=($newsqa $triviaqa $mixed)
for experiment in ${experiments[@]}; do
  echo "Copy model from $squad_model_output to ${experiment}_out"
  cp -r ${squad_model_output}/* ${experiment}_out
  echo "Run ${experiment}.experiment"
  ./run.sh "${experiment}.experiment"
done

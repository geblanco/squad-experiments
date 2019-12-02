#!/bin/bash

# run squad experiment
# ToDo := Should check user confirmation
#   heavy delete operations are carried out.
clean_data_dir() {
  local dir=$1
  if [[ -d $dir ]]; then
    echo "Delete $dir"
    sudo rm -rf $dir
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

run_experiment() {
  echo "###### Starting experiment - $exp"
  start_time=$(date -u +%s)
  clean_data_dir $OUTPUT_DIR
  # run experiment
  echo "Run $exp"
  ./run_experiment.sh $exp
  end_time=$(date -u +%s)
  elapsed=$(python3 -c "print('{:.2f}'.format(($end_time - $start_time)/60.0 ))")
  echo "###### End experiment - $exp - $elapsed minutes"
}

backup() {
  local backup_dir=$1; shift
  local dest_dir=$1; shift
  echo "###### Backup $backup_dir..."
  source ./server_data
  echo "Backup $backup_dir $SRVR_HORACIO_ENV:$dest_dir/"
  rsync -avrzP $backup_dir $SRVR_HORACIO_ENV:$dest_dir/
  echo "###### done"
}

copy_drop_model() {
  if [[ "$TRAIN" == "True" && ! -z $DROP_MODEL ]]; then
    echo "###### Copy model..."
    echo "Copy model $OUTPUT_DIR/checkpoint $drop_base $drop_name"
    drop_base=$(dirname $DROP_MODEL)
    drop_name=$(basename $DROP_MODEL)
    mkdir -p $drop_base
    copy_model $OUTPUT_DIR/checkpoint $drop_base $drop_name
    unset DROP_MODEL
    echo "###### done"
  fi
}

remove_model_data() {
  local dir=$1; shift
  local backup_dir=${dir}_backup
  local to_save=(
  	nbest_predictions.json
  	null_odds.json
  	predictions.json
  	train.log
  	train.run_time
  	train.tf_record
  	eval.tf_record
  )
  mkdir $backup_dir
  for f in ${to_save[@]}; do
    fpath=$dir/$f
    [[ -f $fpath ]] && cp $fpath $backup_dir
  done
  clean_data_dir $dir
  mv $backup_dir/* $dir
  rm -rf $backup_dir
}

# no errors accepted
set -e

source ./server_data
echo "###### Starting experiments $(date)"
total_start_time=$(date -u +%s)
experiments=($@)
for exp in ${experiments[@]}; do
  source $exp
  run_experiment
  if [[ $? -ne 0 ]]; then
    exit $?
  fi
  remote_dest=$(basename `dirname $OUTPUT_DIR`)
  backup $OUTPUT_DIR $remote_dir/$remote_dest
  copy_drop_model
  remove_model_data $OUTPUT_DIR
done
total_end_time=$(date -u +%s)
total_elapsed=$(python3 -c "print('{:.2f}'.format(($total_end_time - $total_start_time)/60.0 ))")
echo "###### End of experiments $(date) ($total_elapsed) minutes"
backup models $remote_models_dir

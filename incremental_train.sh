#!/bin/bash

# train incrementally over the given dataset
clean_data_dir() {
  local dir=$1
  if [[ -d $dir ]]; then
    echo "Delete $dir"
    sudo rm -rf $dir
  fi
  echo "Create $dir"
  mkdir -p $dir
}

clean_multiple_data_dirs() {
  for dir in $@; do
    clean_data_dir $dir
  done
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

sample_dataset() {
  local source=$1
  local train=$2
  local sample_size=$3
  echo "###### Sampling $train"
  python3 scripts/sample_dataset_by_empty.py -d $source -o $train -s $sample_size
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

# no errors accepted
set -e

source ./server_data
echo "###### Starting experiments $(date)"
total_start_time=$(date -u +%s)
experiments=($@)
for exp in ${experiments[@]}; do
  source $exp
  # sample dataset and average over n_steps
  if [[ -z $N_STEPS || -z $SAMPLE_SIZE ]]; then
    echo "Bad experiment without N_STEPS or SAMPLE_SIZE variables"
    continue
  fi
  # folders to save all predictions (small folder) and
  # models with predictions (big folder) 
  AVERAGES_DIR="${OUTPUT_DIR}_averages"
  STEP_MODELS_DIR=${OUTPUT_DIR}_models
  clean_multiple_data_dirs $OUTPUT_DIR $AVERAGES_DIR $STEP_MODELS_DIR
  for step in $(seq $N_STEPS); do
    #### sample dataset and run experiment
      echo "###### Experiment - $exp - step $step"
      sample_dataset $SOURCE_DATASET_FILE $TRAIN_FILE $SAMPLE_SIZE
      run_experiment
      if [[ $? -ne 0 ]]; then
        exit $?
      fi
    #### setup model and storage
      # copy the model for traceability
      if [[ "$TRAIN" == "True" && ! -z $DROP_MODEL ]]; then
        copy_model $OUTPUT_DIR/checkpoint $STEP_MODELS_DIR $(basename $DROP_MODEL)_${step}
        copy_drop_model
      fi
      # copy predictions
      for out_file in nbest_predictions null_odds predictions; do
        cp "$OUTPUT_DIR/${out_file}.json" "$AVERAGES_DIR/${out_file}_${SAMPLE_SIZE}_${step}.json"
        cp "$OUTPUT_DIR/${out_file}.json" "$STEP_MODELS_DIR/${out_file}_${SAMPLE_SIZE}_${step}.json"
      done
    #### backup results
      remote_dest=$(basename `dirname $OUTPUT_DIR`)
      backup $AVERAGES_DIR $remote_dir/$remote_dest
      backup $STEP_MODELS_DIR $remote_dir/$remote_dest
      # delete to save space
      clean_multiple_data_dirs $AVERAGES_DIR $STEP_MODELS_DIR $OUTPUT_DIR
    #### predict on other datasets
    # ToDo := $RUN_AFTER_STEP
  done
done
total_end_time=$(date -u +%s)
total_elapsed=$(python3 -c "print('{:.2f}'.format(($total_end_time - $total_start_time)/60.0 ))")
echo "###### End of experiments $(date) ($total_elapsed) minutes"
backup models $remote_models_dir

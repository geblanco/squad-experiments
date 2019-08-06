#!/bin/bash

# train incrementally over the given dataset
clean_data_dir() {
  for dir in $@; do
    if [[ -d $dir ]]; then
      echo "Delete $dir"
      rm -rf $dir
    fi
    echo "Create $dir"
    mkdir -p $dir
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

# no errors accepted
set -e

experiments=($@)
for exp in ${experiments[@]}; do
  exp_name=$(basename $exp)
  source $exp
  # folders to save all predictions (small folder) and
  # models with predictions (big folder) 
  AVERAGES_DIR="${OUTPUT_DIR}_averages"
  STEP_MODELS_DIR=${OUTPUT_DIR}_models
  clean_data_dir $OUTPUT_DIR $AVERAGES_DIR $STEP_MODELS_DIR
  # run experiment
  echo "Run $exp"
  # sample dataset and average over n_steps
  if [[ -z $N_STEPS || -z $SAMPLE_SIZE ]]; then
    echo "Bad experiment without N_STEPS or SAMPLE_SIZE variables"
    continue
  fi
  for step in $(seq $N_STEPS); do
    echo "############ $exp_name - step $step ############"
    #### sample dataset and run experiment
      echo "Sampling $(basename $TRAIN_FILE)"
      python3 scripts/sample_dataset.py -d $SOURCE_DATASET_FILE -o $TRAIN_FILE -s $SAMPLE_SIZE
      ./run_experiment.sh $exp
      if [[ $? -ne 0 ]]; then
        exit $?
      fi
    #### setup model and storage
      # copy the model for traceability
      copy_model $OUTPUT_DIR/checkpoint $STEP_MODELS_DIR $(basename $DROP_MODEL)_${step}
      if [[ "$TRAIN" == "True" && ! -z $DROP_MODEL ]]; then
        drop_base=$(dirname $DROP_MODEL)
        drop_name=$(basename $DROP_MODEL)
        mkdir -p $drop_base
        echo "Copy model $OUTPUT_DIR/checkpoint $drop_base $drop_name"
        copy_model $OUTPUT_DIR/checkpoint $drop_base $drop_name
        unset DROP_MODEL
      fi
      # copy predictions
      for out_file in nbest_predictions null_odds predictions; do
        cp "$OUTPUT_DIR/${out_file}.json" "$AVERAGES_DIR/${out_file}_${SAMPLE_SIZE}_${step}.json"
        cp "$OUTPUT_DIR/${out_file}.json" "$STEP_MODELS_DIR/${out_file}_${SAMPLE_SIZE}_${step}.json"
      done
      clean_data_dir $OUTPUT_DIR
    #### predict on other datasets
      if [[ ! -z RUN_AFTER_STEP ]]; then
        for run_after_exp in ${RUN_AFTER_STEP[@]}; do
          echo "############ $exp_name - run after $(basename $run_after_exp) ############"
          source $run_after_exp
          [[ ! -d $OUTPUT_DIR ]] && mkdir -p $OUTPUT_DIR
          ./run_experiment.sh $run_after_exp
          if [[ $? -ne 0 ]]; then
            exit $?
          fi
          for out_file in nbest_predictions null_odds predictions; do
            cp "$OUTPUT_DIR/${out_file}.json" "$OUTPUT_DIR/${out_file}_${SAMPLE_SIZE}_${step}.json"
          done
        done
      fi
      source $exp
  done
  #### backup results
    remote_dest=$(basename `dirname $OUTPUT_DIR`)
    echo "Backup $STEP_MODELS_DIR $SRVR_HORACIO_ENV:/data/lihlith/experiments_models/$remote_dest/"
    rsync -avrzP $STEP_MODELS_DIR $SRVR_HORACIO_ENV:/data/lihlith/experiments_models/$remote_dest/
    echo "Backup $AVERAGES_DIR $SRVR_HORACIO_ENV:/data/lihlith/experiments_models/$remote_dest/"
    rsync -avrzP $AVERAGES_DIR $SRVR_HORACIO_ENV:/data/lihlith/experiments_models/$remote_dest/
    # we put `set -e` so exit on error should be enusured, but just in case, to avoid loosing models...
    if [[ $? -ne 0 ]]; then
      exit $?
    fi  
    # delete to save space
    clean_data_dir $AVERAGES_DIR $STEP_MODELS_DIR
done

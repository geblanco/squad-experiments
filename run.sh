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

echo "###### Starting experiments $(date)"
total_start_time=$(date -u +%s)
experiments=($@)
for exp in ${experiments[@]}; do
  echo "###### Starting experiment - $exp"
  start_time=$(date -u +%s)
  source $exp
  clean_data_dir $OUTPUT_DIR
  # run experiment
  echo "Run $exp"
  ./run_experiment.sh $exp
  end_time=$(date -u +%s)
  elapsed=$(python -c "print('{:.2f}'.format(($start_time - $end_time)/60.0 ))")
  echo "###### End experiment - $exp - $elapsed minutes"
  if [[ $? -ne 0 ]]; then
    exit $?
  fi
  echo "###### Backup... - $exp"
  remote_dest=$(basename `dirname $OUTPUT_DIR`)
  echo "Backup $OUTPUT_DIR $SRVR_HORACIO_ENV:/data/lihlith/experiments/$remote_dest/"
  rsync -avrzP $OUTPUT_DIR $SRVR_HORACIO_ENV:/data/lihlith/experiments/$remote_dest/
  if [[ "$TRAIN" == "True" && ! -z $DROP_MODEL ]]; then
    drop_base=$(dirname $DROP_MODEL)
    drop_name=$(basename $DROP_MODEL)
    mkdir -p $drop_base
    echo "Copy model $OUTPUT_DIR/checkpoint $drop_base $drop_name"
    copy_model $OUTPUT_DIR/checkpoint $drop_base $drop_name
    unset DROP_MODEL
  fi
  echo "###### done"
done
total_end_time=$(date -u +%s)
total_elapsed=$(python -c "print('{:.2f}'.format(($total_start_time - $total_end_time)/60.0 ))")
echo "###### End of experiments $(date) ($total_elapsed) minutes"
echo "###### Copying models..."
echo "Backup models $SRVR_HORACIO_ENV:/data/lihlith/"
rsync -avrzP models $SRVR_HORACIO_ENV:/data/lihlith/
echo "######################################"

#!/bin/bash

predict_files=$(ls *.experiment)
preds_dir=`pwd`

var_dump () {
   echo "Variables:
      BATCH_SIZE=$BATCH_SIZE
      SEQ_LENGTH=$SEQ_LENGTH

      BERT_DIR=$BERT_DIR
      SQUAD_DIR=$SQUAD_DIR
      OUTPUT_DIR=$OUTPUT_DIR
      CHECKPOINT=$CHECKPOINT

      TRAIN=$TRAIN
      THRESH=$THRESH

      TRAIN_FILE=$TRAIN_FILE
      PREDICT_FILE=$PREDICT_FILE

      TRAIN_EPOCHS=$TRAIN_EPOCHS" | tee $1
}

for prediction in ${predict_files[@]}; do
  cd ..
  output="${preds_dir}/${prediction%.*}_data"
  mkdir $output
  # source first to do it on root folder
  source "${preds_dir}/${prediction}"
  var_dump "${output}/${prediction}.log"
  ./run.sh "${preds_dir}/${prediction}" | tee "${output}/${prediction}.log"
  # get the results to avoid overwriting
  cp $OUTPUT_DIR/nbest_predictions.json "${output}/nbest_predictions.json"
  cp $OUTPUT_DIR/null_odds.json "${output}/null_odds.json"
  cp $OUTPUT_DIR/predictions.json "${output}/predictions.json"
  cd -
done
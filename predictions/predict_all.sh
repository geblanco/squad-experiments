#!/bin/bash

predict_files=$(ls *.experiment)
for prediction in ${predict_files[@]}; do
  cd ..
  ./run.sh $prediction
  cd -
  source $prediction
  # get the results to avoid overwriting
  cp $OUTPUT_DIR/nbest_predictions.json "${prediction}_nbest_predictions.json"
  cp $OUTPUT_DIR/null_odds.json "${prediction}_null_odds.json"
  cp $OUTPUT_DIR/predictions.json "${prediction}_predictions.json"
done

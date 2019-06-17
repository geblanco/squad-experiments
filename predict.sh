#!/bin/bash

# set -e

if [[ $# -gt 0 ]]; then
  predictions=$@
else
  predictions=$(ls -d predictions/*/ | cut -f2 -d'/')
fi

for prediction in ${predictions[@]}; do
  experiments=$(ls predictions/${prediction}/*.experiment)
  for exp in ${experiments[@]}; do
    output="${exp%.*}_data"
    if [[ -f "${output}/predictions.json" ]]; then
      continue
    fi

    if [[ -d $output ]]; then
      rm -rf $output
    fi
    
    mkdir $output

    source $exp
    ./run_experiment.sh $exp "prediction"
    cp $OUTPUT_DIR/nbest_predictions.json "${output}/nbest_predictions.json"
    cp $OUTPUT_DIR/null_odds.json "${output}/null_odds.json"
    cp $OUTPUT_DIR/predictions.json "${output}/predictions.json"

    # backup data
    [ $BACKUP -eq 0 ] || rsync -avrzP $output $SRVR_ADDR:$SRVR_DEST_DIR
  done
done

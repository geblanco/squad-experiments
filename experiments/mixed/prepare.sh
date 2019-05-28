#!/bin/bash

cwd=`pwd`

BASEDIR=`pwd`/data
# should hold a bert model trained over news QA
NEWS_QA_MODEL_DIR="../newsqa/newsqa_trained_out/"
# should hold the SQuAD converted dataset
TRIVIA_QA_DATA="../triviaqa/data/triviaqa_squad/"

echo "Directories"
echo "BASEDIR=$BASEDIR"
echo "NEWS_QA_MODEL_DIR=$NEWS_QA_MODEL_DIR"
echo "TRIVIA_QA_DATA=$TRIVIA_QA_DATA"

exists() {
  if [[ -d $1 ]]; then
    true
  else
    false
  fi
}

# create an output directory for each experiment
experiment_files=$(ls *.experiment)
for file in ${experiment_files[@]}; do
  output="${file%.*}_out"
  exists $output || mkdir $output
done

# create data dir if necessary
exists $BASEDIR || mkdir $BASEDIR

# check newsqa model and triviaqa data
if [[ ! -d $NEWS_QA_MODEL_DIR || ! -d $TRIVIA_QA_DATA ]]; then
  echo "Either NEWS_QA_MODEL_DIR or TRIVIA_QA_DATA missing!"
fi

cd $cwd

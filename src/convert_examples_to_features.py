from keras_bert import extract_embeddings

from tqdm import tqdm

import numpy as np
import argparse
import json
import sys
import os

FLAGS = None

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', '-d', type=str, required=True,
      help='dataset to analyze')
  parser.add_argument('--model_dir', type=str, required=True, 
      help='directory containing the bert model to use for embedding')
  parser.add_argument('--output_dir', type=str, required=True, 
      help='directory to drop the processed data')
  parser.add_argument('--max_query_length', default=64, type=int,
      help='The maximum number of tokens for the question. Questions longer than '
      'this will be truncated to this length.')
  parser.add_argument('--max_seq_length', default=384, type=int,
      help='The maximum total input sequence length after tokenization. '
      'Sequences longer than this will be truncated, and sequences shorter '
      'than this will be padded.')

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def pad_batch(batch, seq_length):
  batch_length = len(batch)
  if batch_length > seq_length:
    return batch[0:seq_length]
  batch.extend([0]*(seq_length - batch_length))
  return batch

def get_batches(data, batch_size):
  total_amount = len(data)
  n_batches = int(total_amount / batch_size)
  total_processed = 0
  while total_processed < total_amount:
    start = total_processed
    end = start + min(batch_size, total_amount - total_processed)
    total_processed += (end - start)
    yield data[start:end]

def process_data(data, batch_size, seq_length, model_dir, output_dir):
  # get embeddings
  # save as numpy array
  # batch process to avoid memory excess
  total = int(len(data) / batch_size) +1
  for idx, batch in tqdm(enumerate(get_batches(data, batch_size)), total=total):
    batch = [pad_batch(b, seq_length) for b in batch]
    embedded_data = extract_embeddings(model_dir, batch)
    batch_array = np.array(embedded_data)
    output = os.path.join(output_dir, 'batch_{}'.format(idx))
    np.save(output, batch_array)

def main():
  model_dir = FLAGS.model_dir
  data = json.load(open(FLAGS.dataset, 'r'))
  output_dir = FLAGS.output_dir
  contexts = [d['context'] for d in data]
  questions = [d['question'] for d in data]
  context_output_dir = os.path.join(output_dir, 'context')
  question_output_dir = os.path.join(output_dir, 'question')
  os.makedirs(context_output_dir, exist_ok=True)
  os.makedirs(question_output_dir, exist_ok=True)
  print('> Contexts')
  process_data(contexts, batch_size=512, seq_length=FLAGS.max_seq_length, 
      model_dir=model_dir, output_dir=context_output_dir)
  print('> Questions')
  process_data(questions, batch_size=512, seq_length=FLAGS.max_query_length,
      model_dir=model_dir, output_dir=question_output_dir)

if __name__ == '__main__':
  FLAGS = parse_args()
  main()

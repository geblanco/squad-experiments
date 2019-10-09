from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.models import Sequential
from keras.utils import multi_gpu_model
from keras import preprocessing

from tensorflow.python.client import device_lib

from typing import NamedTuple

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import argparse
import json
import bz2
import sys
import os

## for replicability of results
np.random.seed(1)
tf.set_random_seed(2)

FLAGS = None

class ModelParams(NamedTuple):
  embeddings_size: int = 0
  max_words: int = 0
  lstm_hidden_size: int = 0
  regularization: float = 0.0
  dropout: float = 0.0
  lrate: float = 0.0

class Example(NamedTuple):
  id: int = 0
  question_text: str = ''
  context_text: str = ''
  doc_tokens: list = []
  answers: list = []
  correct: int = -1

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_file', default=None, required=True, type=str,
    help='Json file for training. E.g., train-ensemble-qa-v1.0.json')
  parser.add_argument('--validate_file', default=None, required=True, type=str,
      help='Json file for validation. E.g., dev-ensemble-qa-v1.0.json')
  parser.add_argument('--predict_file', default=None, required=True, type=str,
      help='Json for predictions. E.g., test-ensemble-qa-v1.1.json')
  parser.add_argument('--do_train', default=False, help='Whether to run training.')
  parser.add_argument('--do_predict', default=False, help='Whether to run eval on the test set.')
  parser.add_argument('--batch_size', default=128, type=int,
      help='Total batch size for training.')
  parser.add_argument('--learning_rate', default=5e-5, type=float,
      help='The initial learning rate for Adam.')
  parser.add_argument('--num_train_epochs', default=20, type=int,
      help='Total number of training epochs to perform.')
  parser.add_argument('--output_dir', default=None, required=True,
      help='The output directory where figures and model will be written.')
  parser.add_argument('--max_seq_length', default=384, type=int,
      help='The maximum total input sequence length after tokenization. '
      'Sequences longer than this will be truncated, and sequences shorter '
      'than this will be padded.')
  parser.add_argument('--max_words_length', default=10000, type=int, 
      help='The maximum number of words accepted in the dictionary.')
  parser.add_argument('--lstm_hidden_size', default=128, type=int, required=True,
      help='The size of the internal representation on lstm layers.')
  parser.add_argument('--embeddings_file', default=None, type=str, required=True, 
      help='Embeddings file to load on the model. E.g., glove.6B.50d.txt.bz2')
  parser.add_argument('--embeddings_size', default=50, type=int, required=True, 
      help='Embeddings file to load on the model. E.g., glove.6B.50d.txt.bz2')
  if len(sys.argv) < 2:
    parser.print_help()
    sys.exit(0)
  return parser.parse_args()

def get_num_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x for x in local_device_protos 
                    if x.device_type == 'GPU'])

def is_whitespace(c):
  if c == ' ' or c == '\t' or c == '\r' or c == '\n' or ord(c) == 0x202F:
    return True
  return False

def get_doc_tokens(text):
  doc_tokens = []
  prev_is_whitespace = True
  for c in text:
    if is_whitespace(c):
      prev_is_whitespace = True
    else:
      if prev_is_whitespace:
        doc_tokens.append(c)
      else:
        doc_tokens[-1] += c
    prev_is_whitespace = False
  return doc_tokens

def read_examples(input_file, is_training):
  data = json.load(open(input_file, 'r'))
  examples = []
  for entry in data:
    # when training, skip examples with no correct system
    if is_training and entry['correct'] == -1:
      continue

    example = Example(
        id=entry['id'],
        question_text=entry['question'],
        context_text=entry['context'],
        doc_tokens=get_doc_tokens(entry['context']),
        answers=entry['answers'],
        correct=entry['correct'])
    examples.append(example)

  return examples

def get_texts_and_labels_from_examples(examples):
  texts = []
  labels = []
  for example in examples:
    text = example.context_text + example.question_text + ' '.join(example.answers)
    texts.append(text)
    labels.append(example.correct)
  return texts, labels

# def train_dev_test_split(texts, labels):
#   # sklearn only splits into two sets, divide twice to get the three sets (60, 10, 20)
#   train_texts, train_labels, dev_and_test_texts, dev_and_test_labels = train_test_split(
#       texts, labels, test_size=0.4, random_state=42)
#   dev_texts, dev_labels, test_texts, test_labels = train_test_split(
#       dev_and_test_texts, dev_and_test_labels, test_size=0.5, random_state=42)
#   return train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels

def get_padded_sequences(tokenizer, texts, max_seq):
  sequences = tokenizer.texts_to_sequences(texts)
  padded = preprocessing.sequence.pad_sequences(sequences, maxlen=max_seq)
  return padded

def prepare_set(examples, max_words, max_seq, tokenizer, fit=False):
  texts, labels = get_texts_and_labels_from_examples(examples)
  if fit:
    # Build the word index (dictionary)
    tokenizer.fit_on_texts(texts)

  # Get data as a lists of integers and pad, 2D integer tensor of shape `(samples, max_seq)`
  x = get_padded_sequences(tokenizer, texts, max_seq)

  return x, labels

def parse_embeddings(file):
  embeddings = {}
  lines = file.read()
  file.close()
  lines = lines.decode('utf8')
  for line in lines.split('\n'):
    vec = line.split(' ')
    word = vec[0]
    coefs = np.asarray(vec[1:], dtype='float32')
    embeddings[word] = coefs
  return embeddings  

def build_embedding_matrix(file, threshold, dim, word_index):
  # ToDo := Nof misrepresented words
  max_words = 400000 if threshold <= 0 else min(threshold, 400000)
  embeddings = parse_embeddings(file)
  matrix = np.zeros((max_words, dim))
  for word, i in word_index.items():
    if i < max_words:
      embedding_vector = embeddings.get(word)
      if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        matrix[i] = embedding_vector
  return matrix

def build_model(embedding_matrix, model_params):
  # Build model on CPU and distribute to all gpus, the gradients will come back
  # to cpu to be calculated
  with tf.device('/cpu:0'):
    model = Sequential()
    # 1. Define and add Embedding layer to the model
    model.add(Embedding(model_params.max_words, model_params.embeddings_size, 
        mask_zero=True))
    # After the Embedding layer, 
    # our activations have shape `(batch_size, max_seq, embeddings_size)`.
    # 2. Define and add LSTM layer to the model.
    model.add(Bidirectional(LSTM(model_params.lstm_hidden_size,
        kernel_regularizer=l2(model_params.regularization))))
    # 3. Avoid overfitting with dropout
    model.add(Dropout(model_params.dropout))
    # 4. Define and add Dense layer to the model
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False

    adam = Adam(lr=model_params.lrate)

  parallel_model = multi_gpu_model(model, gpus=get_num_available_gpus())
  parallel_modelmodel.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
  return model, parallel_model

def train_model(model, epochs, batch_size, x_train, y_train, x_dev, y_dev):
  early_stop = EarlyStopping(monitor='accuracy', patience=2)
  hist = model.fit(
      x_train,
      y_train,
      epochs=epochs,
      batch_size=batch_size,
      validation_data=(x_dev, y_dev),
      verbose=1,
      callbacks=[early_stop])
  return hist

def plot_results(metrics_dict, keys, plot_name, plot=False, out_file=None):
  for key in keys:
    plt.plot(metrics_dict['loss'])

  plot_type = 'loss' if 'loss' in keys else 'accuracy' 
  plt.title('model ' + plot_type)
  plt.ylabel(plot_type)
  plt.xlabel('epochs')
  plt.legend(['train', 'dev'], loc='upper left')
  if plot:
    plt.show()
  if out_file is not None:
    plt.savefig(out_file)

def save_model(model, model_structure_file, model_weigths_file):
  # serialize model to JSON
  model_json = model.to_json()
  json.dump(fp=open(model_structure_file, 'w'), obj=model_json)
  # serialize weights to HDF5
  model.save_weights(model_weigths_file)

def restore_model(model_structure_file, model_weigths_file):
  # load json and create model
  with open(model_structure_file, 'r') as json_file:
    loaded_model_json = json_file.read()
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights(model_weigths_file)
  return loaded_model

def main():
  tf.gfile.MakeDirs(FLAGS.output_dir)
  # params
  max_words = FLAGS.max_words_length
  max_seq = FLAGS.max_seq_length
  lstm_hidden_size = FLAGS.lstm_hidden_size
  epochs = FLAGS.num_train_epochs
  batch_size = FLAGS.batch_size
  embeddings_size = FLAGS.embeddings_size

  # Create a tokenize that takes the 10000 most common words
  tokenizer = preprocessing.text.Tokenizer(num_words=max_words)

  train_examples = read_examples(FLAGS.train_file, is_training=True)
  dev_examples = read_examples(FLAGS.validate_file, is_training=True)
  test_examples = read_examples(FLAGS.predict_file, is_training=False)

  shuffle(train_examples)
  shuffle(dev_examples)
  shuffle(test_examples)

  x_train, y_train = prepare_set(train_examples, max_words, max_seq, tokenizer=tokenizer, fit=True)
  x_dev, y_dev = prepare_set(train_examples, max_words, max_seq, tokenizer=tokenizer)
  x_test, y_test = prepare_set(train_examples, max_words, max_seq, tokenizer=tokenizer)

  print('Shape of the training set (nb_examples, vector_size): {}'.format(x_train.shape))
  print('Shape of the validation set (nb_examples, vector_size): {}'.format(x_dev.shape))
  print('Shape of the test set (nb_examples, vector_size): {}'.format(x_test.shape))

  # Read input embeddings
  word_index = tokenizer.word_index
  with bz2.open(FLAGS.embeddings_file) as embsfile:
    embedding_matrix = build_embedding_matrix(embsfile, threshold=max_words, dim=embeddings_size, word_index=word_index)

  print('Embeddings shape: {}'.format(embedding_matrix.shape))

  model_params = ModelParams(
      embeddings_size=embeddings_size,
      max_words=max_words,
      lstm_hidden_size=lstm_hidden_size,
      regularization=0.01,
      dropout=0.2,
      lrate=FLAGS.learning_rate)

  model_structure_file = os.path.join(FLAGS.output_dir, 'model.json')
  model_weigths_file = os.path.join(FLAGS.output_dir, 'model.h5')

  if FLAGS.do_train:
    model, parallel_model = build_model(embedding_matrix, model_params)
    # train on parallel model
    metrics = train_model(parallel_model, epochs, batch_size, x_train, y_train, x_dev, y_dev)
    plot_results(metrics.history, ['loss', 'val_loss'], plot_name='model loss',
      out_file=os.path.join(FLAGS.output_dir, 'train_loss.png'))
    plot_results(metrics.history, ['accuracy', 'val_accuracy'], plot_name='model accuracy',
        out_file=os.path.join(FLAGS.output_dir, 'train_accuracy.png'))
    # save unified model
    save_model(model, model_structure_file, model_weigths_file)
  else:
    # if not training, restore model from file
    model = restore_model(model_structure_file, model_weigths_file)

  if FLAGS.do_predict:
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Final Accuracy: ', score[1])

if __name__ == '__main__':
  FLAGS = parse_args()
  main()

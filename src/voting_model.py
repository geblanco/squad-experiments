from __future__ import absolute_import, division, print_function, unicode_literals
from typing import NamedTuple

import tokenization
import tensorflow as tf
import numpy as np
import argparse

OPTS = None

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', '-d', type=str, help='dataset to analyze')

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
  lang_tokenizer.fit_on_texts(lang)
  tensor = lang_tokenizer.texts_to_sequences(lang)
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
  return tensor, lang_tokenizer

from matplotlib import pyplot as plt
import json, argparse, os, sys, re

args = None

class InputFeatures(NamedTuple):
  """A single set of features of data."""

  unique_id: int = 0
  question_ids: list = []
  context_ids: list = []
  answers_ids: list = []

class Example(NamedTuple):
  id: int = 0
  question_text: str = ''
  context_text: str = ''
  doc_tokens: list = []
  answers: list = []
  correct: int = -1

class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

  def __init__(self, filename):
    self.filename = filename
    self.num_features = 0
    self._writer = tf.python_io.TFRecordWriter(filename)

  def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    features = collections.OrderedDict()
    features["unique_ids"] = create_int_feature([feature.unique_id])
    features["question_ids"] = create_int_feature(feature.question_ids)
    features["context_ids"] = create_int_feature(feature.context_ids)
    features["answers_ids"] = create_int_feature(feature.answers_ids)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()

class DatasetIO(object):
  def __init__(self, input_file, tf_record_file, tokenizer, max_seq_length):
    self.input_file = input_file
    self.tf_record_file = tf_record_file
    self.tokenizer = tokenizer
    self.max_seq_length = max_seq_length

  def _read_examples(self):
    def is_whitespace(c):
      if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
      return False

    data = json.load(open(self.input_file, 'r'))
    examples = []
    for entry in data:
      doc_tokens = []
      for c in entry['context']:
        if is_whitespace(c):
          prev_is_whitespace = True
        else:
          if prev_is_whitespace:
            doc_tokens.append(c)
          else:
            doc_tokens[-1] += c
        prev_is_whitespace = False

      example = Example(
          id=entry['id'],
          question_text=entry['question'],
          context_text=entry['context'],
          doc_tokens=doc_tokens,
          answers=entry['answers'],
          correct=entry['correct'])
      examples.append(example)

    return examples

  def _convert_examples_to_features(self, examples, output_fn):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    for (example_index, example) in enumerate(examples):
      query_tokens = self.tokenizer.tokenize(example.question_text)

      if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[0:max_query_length]

      tok_to_orig_index = []
      orig_to_tok_index = []
      all_doc_tokens = []
      for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = self.tokenizer.tokenize(token)
        for sub_token in sub_tokens:
          tok_to_orig_index.append(i)
          all_doc_tokens.append(sub_token)

      tok_start_position = None
      tok_end_position = None
      if is_training and example.is_impossible:
        tok_start_position = -1
        tok_end_position = -1
      if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
          tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
          tok_end_position = len(all_doc_tokens) - 1
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, self.tokenizer,
            example.orig_answer_text)

      # The -3 accounts for [CLS], [SEP] and [SEP]
      max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

      # We can have documents that are longer than the maximum sequence length.
      # To deal with this we do a sliding window approach, where we take chunks
      # of the up to our max length with a stride of `doc_stride`.
      _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
          "DocSpan", ["start", "length"])
      doc_spans = []
      start_offset = 0
      while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
          length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
          break
        start_offset += min(length, doc_stride)

      for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
          tokens.append(token)
          segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        for i in range(doc_span.length):
          split_token_index = doc_span.start + i
          token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

          is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                 split_token_index)
          token_is_max_context[len(tokens)] = is_max_context
          tokens.append(all_doc_tokens[split_token_index])
          segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
          input_ids.append(0)
          input_mask.append(0)
          segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        start_position = None
        end_position = None
        if is_training and not example.is_impossible:
          # For training, if our document chunk does not contain an annotation
          # we throw it out, since there is nothing to predict.
          doc_start = doc_span.start
          doc_end = doc_span.start + doc_span.length - 1
          out_of_span = False
          if not (tok_start_position >= doc_start and
                  tok_end_position <= doc_end):
            out_of_span = True
          if out_of_span:
            start_position = 0
            end_position = 0
          else:
            doc_offset = len(query_tokens) + 2
            start_position = tok_start_position - doc_start + doc_offset
            end_position = tok_end_position - doc_start + doc_offset

        if is_training and example.is_impossible:
          start_position = 0
          end_position = 0

        # if example_index < 20:
        #   tf.logging.info("*** Example ***")
        #   tf.logging.info("unique_id: %s" % (unique_id))
        #   tf.logging.info("example_index: %s" % (example_index))
        #   tf.logging.info("doc_span_index: %s" % (doc_span_index))
        #   tf.logging.info("tokens: %s" % " ".join(
        #       [tokenization.printable_text(x) for x in tokens]))
        #   tf.logging.info("token_to_orig_map: %s" % " ".join(
        #       ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
        #   tf.logging.info("token_is_max_context: %s" % " ".join([
        #       "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
        #   ]))
        #   tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #   tf.logging.info(
        #       "input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #   tf.logging.info(
        #       "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #   if is_training and example.is_impossible:
        #     tf.logging.info("impossible example")
        #   if is_training and not example.is_impossible:
        #     answer_text = " ".join(tokens[start_position:(end_position + 1)])
        #     tf.logging.info("start_position: %d" % (start_position))
        #     tf.logging.info("end_position: %d" % (end_position))
        #     tf.logging.info(
        #         "answer: %s" % (tokenization.printable_text(answer_text)))

        feature = InputFeatures(
            unique_id=unique_id,
            example_index=example_index,
            doc_span_index=doc_span_index,
            tokens=tokens,
            token_to_orig_map=token_to_orig_map,
            token_is_max_context=token_is_max_context,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            start_position=start_position,
            end_position=end_position,
            is_impossible=example.is_impossible)

        # Run callback
        output_fn(feature)

        unique_id += 1

  def _encode_record(self):
  
  def _decode_record(self):

  def get_tf_dataset(self):

  def write_examples_as_features(self):
    examples = self._read_examples()

    f_writer = FeatureWriter(filename=self.tf_record_file)
    features = []

    def append_feature(feature):
      features.append(feature)
      f_writer.process_feature(feature)

    self._convert_examples_to_features(
        examples=examples,
        output_fn=append_feature)
    f_writer.close()

    return examples, features


def load_data(input_file):
  

def main():
  tf.logging.set_verbosity(tf.logging.INFO)
  data = json.load(open(OPTS.dataset, 'r'))

if __name__ == '__main__':
  OPTS = parse_args()
  main()

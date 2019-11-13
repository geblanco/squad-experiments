from sklearn.utils import shuffle
from tqdm import tqdm
import argparse
import random
import json
import sys
import os

FLAGS = None
# Tolerance in number of questions
TOTAL_TOLERANCE = 50
EMPTY_TOLERANCE = 50
TOPIC_COUNT = {}

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--datasets', dest='datasets', help='Train and dev'
      'datasets', required=True, type=str, nargs='*')
  parser.add_argument('-o', '--output', dest='output', required=True, type=str,
    help='file to put the sampled dataset')
  parser.add_argument('-p', '--proportions', action='store', nargs="*",
    help='proportions of empty answers to split the dataset', 
    required=True, type=float)
  return parser.parse_args()

def count_topic_ans(topic):
  topic_count = { 'has_ans': 0, 'no_ans': 0 }
  for p in topic['paragraphs']:
    for qa in p['qas']:
      topic_count['has_ans'] += int(bool(qa['answers']))
      topic_count['no_ans'] += int(not bool(qa['answers']))
  topic_count['total'] = topic_count['has_ans'] + topic_count['no_ans']
  return topic_count

def count_all_topics_ans(dataset):
  global TOPIC_COUNT
  topics = {}
  for article in dataset:
    title = article['title']
    if TOPIC_COUNT.get(title, None) is None:
      TOPIC_COUNT[title] = count_topic_ans(article)
    topics[title] = TOPIC_COUNT[title]
  return topics

def count_total_ans(dataset):
  curr_count = count_all_topics_ans(dataset)
  total = sum([topic['total'] for topic in curr_count.values()])
  empty = sum([topic['no_ans'] for topic in curr_count.values()])
  return total, empty

def check_boundaries(total, empty, max_total, max_empty):
  acceptable = False
  if (max_total > total) or abs(max_total - total) <= TOTAL_TOLERANCE:
    if (max_empty > empty) or abs(max_empty - empty) <= EMPTY_TOLERANCE:
      acceptable = True
  return acceptable

def check_add(split_set, elem, sizes):
  total, empty = count_total_ans(split_set)
  elem_count = count_topic_ans(elem)
  inclusion_total = total + elem_count['total']
  inclusion_empty = empty + elem_count['no_ans']
  max_total = sizes['total']
  max_empty = sizes['empty']
  return check_boundaries(inclusion_total, inclusion_empty, max_total, max_empty)

def check_end(split_set, sizes):
  total, empty = count_total_ans(split_set)
  max_total = sizes['total']
  max_empty = sizes['empty']
  end = abs(max_total - total) <= TOTAL_TOLERANCE and \
      abs(max_empty - empty) <= EMPTY_TOLERANCE
  return end

def add_splits_to_dataset(train_set, topics, topic_idx, split_sizes):
  done = False
  idx = topic_idx
  n_topics = len(topics)
  sizes = split_sizes['train']
  while not done and idx < n_topics:
    elem = topics[idx]
    if check_add(train_set, elem, sizes):
      train_set.append(elem)
      add_splits_to_dataset(train_set, topics, idx+1, split_sizes)
      done = check_end(train_set, sizes)
      if not done:
        train_set.remove(elem)
    idx += 1
  return train_set

def split_dataset(dataset, split_sizes):
  train_set= []
  idx = 0
  add_splits_to_dataset(
    train_set,
    dataset,
    idx,
    split_sizes)
  train_set_keys = [topic['title'] for topic in train_set]
  dev_set = [topic for topic in dataset \
      if topic['title'] not in train_set_keys]
  return train_set, dev_set

def get_proportions(datasets):
  lens = []
  proportions = []
  for dataset in datasets:
    total, _ = count_total_ans(dataset)
    lens.append(total)
  total = sum(lens)
  proportions = [l/total for l in lens]
  return proportions

def setup_sizes(total_proportions, empty_proportions, total_len, empty_len):
  train_total = round(total_proportions[0] * total_len)
  dev_total = total_len - train_total
  train_empty = round(empty_proportions[0] * train_total)
  dev_empty = empty_len - train_empty
  print(f'Setup sizes {total_proportions} {empty_proportions} {total_len} {empty_len}')
  split_sizes = {
    'train': { 'total': train_total, 'empty': train_empty },
    'dev': { 'total': dev_total, 'empty': dev_empty }
  }
  return split_sizes

def sort_dataset_by_topic_score(dataset, topics):
  def sort_topic(topic):
    total = topics[topic]['total']
    has_ans_score = (topics[topic]['has_ans']*100)/ total
    no_ans_score = (topics[topic]['no_ans']*100)/total
    return (has_ans_score **2) + (no_ans_score **2)

  def sort_dataset(item):
    return topic_keys.index(item['title'])

  topic_keys = list(topics.keys())
  topic_keys.sort(key=sort_topic)
  dataset.sort(key=sort_dataset)
  return dataset

def merge_datasets(datasets):
  ret = datasets[0].copy()
  for dataset in datasets[1:]:
    ret.extend(dataset)
  return ret

def save_dataset(path, dataset, **kwargs):
  with open(path, 'w') as f:
    json.dump(fp=f, obj={ 'data': dataset, **kwargs })

def main(dataset_files, proportions):
  global TOPIC_COUNT
  datasets = []
  for file in dataset_files:
    dataset = json.load(open(file , 'r'))['data']
    datasets.append(dataset)
  full_dataset = merge_datasets(datasets)
  shuffle(full_dataset)
  topic_count = count_all_topics_ans(full_dataset)
  total, empty = count_total_ans(full_dataset)
  print(f'Total dataset len {total}, empty {empty}')
  total_proportions = get_proportions(datasets)
  sizes = setup_sizes(total_proportions, proportions, total, empty)
  print(f'Sizes {sizes}')
  # sort from min to max difference between total and emtpy answers
  full_dataset = sort_dataset_by_topic_score(full_dataset, topic_count)
  train, dev = split_dataset(full_dataset, sizes)
  train_name = os.path.join(FLAGS.output, f'squad-train-{proportions[0]}.json')
  dev_name = os.path.join(FLAGS.output, f'squad-dev-{proportions[1]}.json')
  train_total, train_empty = count_total_ans(train)
  dev_total, dev_empty = count_total_ans(dev)
  print(f'End, train count {(train_total, train_empty)}: {train_empty/train_total}, saving to {train_name}') 
  print(f'End, dev count {(dev_total, dev_empty)}: {dev_empty/dev_total}, saving to {dev_name}') 
  save_dataset(train_name, train)
  save_dataset(dev_name, dev)

if __name__ == '__main__':
  FLAGS = parse_args()
  if any([p >= 1.0 for p in FLAGS.proportions]) or \
    any([p < 0 for p in FLAGS.proportions]):
    raise ValueError('Proportions must be between 0 and 1')
  if len(FLAGS.proportions) != len(FLAGS.datasets):
    raise ValueError('The number of proportions and datasets given must match')
  main(FLAGS.datasets, FLAGS.proportions)


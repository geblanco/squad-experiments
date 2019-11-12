from tqdm import tqdm
import argparse
import random
import json
import sys
import os

FLAGS = None
TOLERANCE = 0.005
TOPIC_COUNT = None

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

def count_all_topics_ans_first(dataset):
  topics = {}
  for article in tqdm(dataset):
    title = article['title']
    topics[title] = count_topic_ans(article)
  return topics

def count_all_topics_ans(dataset):
  global TOPIC_COUNT
  topics = {}
  for article in dataset:
    title = article['title']
    topics[title] = TOPIC_COUNT[title]
  return topics

def count_total_ans(dataset):
  curr_count = count_all_topics_ans(dataset)
  total = sum([topic['total'] for topic in curr_count.values()])
  empty = sum([topic['no_ans'] for topic in curr_count.values()])
  return total, empty

def check_boundaries(total, empty, max_total, max_empty):
  acceptable = False
  if (max_total > total) or abs(max_total - total) <= round(max_total * TOLERANCE):
    if (max_empty > empty) or abs(max_empty - empty) <= round(max_empty * TOLERANCE):
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
  return abs(max_total - total) <= round(max_total * TOLERANCE) and \
      abs(max_empty - empty) <= round(max_empty * TOLERANCE)

def add_splits_to_dataset(train_set, topics, topic_idx, split_sizes):
  done = False
  idx = topic_idx
  n_topics = len(topics)
  sizes = split_sizes['train']
  if topic_idx == 0:
    pbar = tqdm(total=n_topics)
  while not done and idx < n_topics:
    elem = topics[idx]
    if check_add(train_set, elem, sizes):
      train_set.append(elem)
      add_splits_to_dataset(train_set, topics, idx+1, split_sizes)
      done = check_end(train_set, sizes)
      if not done:
        train_set.remove(elem)
    if topic_idx == 0:
      pbar.update()
    idx += 1
  if topic_idx == 0:
    pbar.close()
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
    lens.append(len(dataset))
  total = sum(lens)
  proportions = [l/total for l in lens]
  return proportions

def setup_sizes(total_proportions, empty_proportions, total_len, empty_len):
  train_total = round(total_proportions[0] * total_len)
  dev_total = total_len - train_total
  total_empty_proportion = empty_len/total_len
  train_empty = round(total_empty_proportion * train_total)
  dev_empty = empty_len - train_empty
  print(f'Setup sizes {total_proportions} {empty_proportions} {total_len} {empty_len}')
  split_sizes = {
    'train': { 'total': train_total, 'empty': train_empty },
    'dev': { 'total': dev_total, 'empty': dev_empty }
  }
  return split_sizes

def merge_datasets(datasets):
  for dataset in datasets[1:]:
    datasets[0].extend(dataset)
  return datasets[0]

def save_dataset(path, dataset):
  with open(path, 'w') as f:
    json.dump(fp=f, obj={ 'data': dataset })

def main(dataset_files, proportions):
  global TOPIC_COUNT
  datasets = []
  for file in dataset_files:
    dataset = json.load(open(file , 'r'))['data']
    datasets.append(dataset)
  total_proportions = get_proportions(datasets)
  full_dataset = merge_datasets(datasets)
  TOPIC_COUNT = count_all_topics_ans_first(full_dataset)
  total, empty = count_total_ans(full_dataset)
  print(f'Total dataset len {total}, empty {empty}')
  sizes = setup_sizes(total_proportions, proportions, total, empty)
  print(f'Sizes {sizes}')
  train, dev = split_dataset(full_dataset, sizes)
  print(f'End, train count {count_total_ans(train)}') 
  print(f'End, dev count {count_total_ans(dev)}') 
  train_name = os.path.join(output, f'squad-train-{proportions[0]}.json'),
  dev_name = os.path.join(output, f'squad-dev_-{proportions[0]}.json'),
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


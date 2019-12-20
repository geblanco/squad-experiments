import random, argparse
import json, sys, math, os

flags = None
EMPTY_ANS_KEY = 1
HAS_ANS_KEY = 2

THRESHOLD = 0.001
REMOVE_AMOUNT = 0.05

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--dataset', dest='dataset', help='dataset to'
      'sample', required=True, type=str)
  parser.add_argument('-o', '--output', dest='output', help='file to put the'
      'sampled dataset', required=True, type=str)
  parser.add_argument('-s', '--sample', dest='sample', help='percentage of empty'
      'answers to keep in the sampled dataset', required=True, type=int)
  return parser.parse_args()

def make_qid_to_has_ans(dataset):
  qid_to_has_ans = {}
  for article in dataset:
    for p in article['paragraphs']:
      for qa in p['qas']:
        qid_to_has_ans[qa['id']] = bool(qa['answers'])
  return qid_to_has_ans

def make_qid_to_dataset(dataset, qids):
  for article in dataset:
    pas = []
    for p in article['paragraphs']:
      qas = []
      for qa in p['qas']:
        qid = qa['id']
        if qid in qids:
          qas.append(qa)
      p['qas'] = qas
      if len(qas):
        pas.append(p)
    article['paragraphs'] = pas
  return dataset

# return whether the percentage is whithin target (True/False) and the
# approximation direction (-1/1)
def check_empty_percentage(qid_to_has_ans, qids, target):
  total = len(qids)
  numof_no_ans = sum([1 for q in qids if not qid_to_has_ans[q]])
  perc_no_ans = numof_no_ans / total
  check = False
  side = 0
  print(f'P {perc_no_ans}, T {target}, TH, {THRESHOLD}')
  if abs(perc_no_ans - target) <= THRESHOLD:
    check = True
  if perc_no_ans < target:
    side = -1
  elif perc_no_ans > target:
    side = 1
  return check, side

def remove_ans(qids, to_remove):
  # qids is sorted, empties from the left, has ans from the right
  amount = math.ceil(REMOVE_AMOUNT * len(qids) / 100)
  if to_remove == EMPTY_ANS_KEY:
    return qids[amount:]
  return qids[:(amount * -1)]

def remove_no_ans(qid_to_has_ans):
  return [q for q, v in qid_to_has_ans.items() if v]

def sample_dataset(qid_to_has_ans, qids, target):
  if target == 0:
    return remove_no_ans(qid_to_has_ans)
  check, direction = check_empty_percentage(qid_to_has_ans, qids, target)
  if check:
    return qids
  current_direction = direction
  to_remove = EMPTY_ANS_KEY
  if direction == -1:
    to_remove = HAS_ANS_KEY
  while not check and current_direction == direction:
    # remove answer and check again
    qids = remove_ans(qids, to_remove)
    check, direction = check_empty_percentage(qid_to_has_ans, qids, target)

  if not check and current_direction != direction:
    # change in direction but not whithin boundaries means we have jumped to
    # other side of the slope, too small THRESHOLD or too few samples
    print('Warning: Unable to target the wanted percentage, either the'
        'THRESHOLD is too small or there are too few samples!')
  return qids

def setup_dataset(qid_to_has_ans):
  # shuffle the dataset to randomize choices,
  # sort based on has/no ans to be able to remove from the left/right
  dataset_qids = list(qid_to_has_ans.keys())
  random.shuffle(dataset_qids)
  sorted_qids = sorted(dataset_qids, key=lambda qid: qid_to_has_ans[qid])
  return sorted_qids, { qid: qid_to_has_ans[qid] for qid in sorted_qids }

def main(flags):
  dataset = json.load(open(flags.dataset, 'r'))['data']
  qid_to_has_ans = make_qid_to_has_ans(dataset)
  sorted_qids, sorted_qid_to_has_ans = setup_dataset(qid_to_has_ans)
  final_qids = sample_dataset(qid_to_has_ans, sorted_qids, flags.sample)
  dataset = make_qid_to_dataset(dataset, final_qids)
  print(f'Final number of questions {len(dataset)}')
  json.dump(fp=open(flags.output, 'w'), obj={ 'data': dataset })

if __name__ == '__main__':
  flags = parse_args()
  flags.sample /= 100.0
  main(flags)


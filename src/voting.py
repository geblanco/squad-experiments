from collections import Counter
import argparse
import json
import sys

'''
Given a dataset with questions, gold standard agent with correct answer
and set of answers by each agent, calculate the score if a voting scheme would
have been used
'''

OPTS = None

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', '-d', type=str, help='dataset to analyze')
  parser.add_argument('--voting', '-v', type=str, dest='voting_scheme',
      choices=('majority', 'minority'), default='majority', help='the voting scheme to use')

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def count_answers(answers):
  return Counter(answers)

def sort_dict_by_value(to_sort_dict, order):
  reverse = (order == 'max')
  return dict(sorted(to_sort_dict.items(), key=lambda x: x[1],
    reverse=reverse))

def majority_voting(dataset):
  total_score = 0.0
  for data_point in dataset:
    correct_system_index = data_point['correct']
    if correct_system_index == -1:
      # no system was able to answer correctly, skip
      continue

    answer_count_dict = count_answers(data_point['answers'])
    sorted_ac = sort_dict_by_value(answer_count_dict, order='max')

    x_system_answer = list(sorted_ac.keys())[0]
    y_system_answer = data_point['answers'][correct_system_index]
    if y_system_answer == x_system_answer:
      total_score += 1

  return 100 * (total_score / len(dataset))

def main(dataset, voting_scheme):
  if voting_scheme == 'majority':
    print('Score by {} voting: {}'.format(voting_scheme,
      majority_voting(dataset)))

if __name__ == '__main__':
  OPTS = parse_args()
  dataset = json.load(open(OPTS.dataset, 'r'))
  main(dataset, OPTS.voting_scheme)

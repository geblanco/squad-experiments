# measure f measure for empty answers

import argparse
import json, sys

OPTS = None

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-s', '--scores', action='store', dest='scores', required=True,
                      type=str, nargs="*", help='list of scores files to get the upper bound')
  args = parser.parse_args()
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def upper_bound(scores):
  total_correct = 0
  for question_set in zip(*[s.items() for s in scores]):
    # [(id, score), (id, score), ...] -> [(id, id, ...), (score, score, ...)]
    ids, score_set = list(zip(*question_set))
    if len(set(ids)) > 1:
      raise ValueError('Different question ids', ids, score_set)
    total_correct += int(max(score_set))
  return { 'upper_bound': 100.0 * (total_correct / len(scores[0])) }

def main():
  scores = [json.load(open(p, 'r')) for p in OPTS.scores]
  print(json.dumps(upper_bound(scores), indent=2))

if __name__ == '__main__':
  OPTS = parse_args()
  main()
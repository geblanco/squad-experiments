# measure f measure for empty answers

from collections import namedtuple
import argparse
import json, sys, os

OPTS = None

Example = namedtuple('Example', ['id', 'context', 'question', 'systems', 'correct'])

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-s', '--sweep', dest='sweep', required=False, type=str,
                      help='directory with results to sweep')
  parser.add_argument('-e', '--exact_scores', action='store', dest='scores',
                      required=False, type=str, nargs="*", 
                      help='list of scores files to get the upper bound')
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
  return 100.0 * (total_correct / len(scores[0]))

def get_scores_from_dir(scores_dir):
  scores_dict = {}
  result_dirs = os.listdir(scores_dir)
  for result_dir in result_dirs:
    scores_path = os.path.join(scores_dir, result_dir, 'exact_scores.json')
    if os.path.exists(scores_path):
      scores_dict[result_dir] = json.load(open(scores_path, 'r'))
  return scores_dict

def sweep(scores_dir):
  max_score = 0.0
  keep_scores = {}
  scores_dict = get_scores_from_dir(scores_dir)
  print('Got scores', scores_dict.keys())
  for score_name, score_value in scores_dict.items():
    score = upper_bound(list(keep_scores.values()) + [score_value])
    if score > max_score:
      keep_scores[score_name] = score_value
      max_score = score
  return list(keep_scores.keys()), max_score

def scores_names_to_id(scores_names):
  return [i for i, _ in enumerate(scores_names)]

def squad_to_question_and_context(original_dataset):
  output = {}
  for entry in original_dataset:
    for paragraph in entry['paragraphs']:
      for qa in paragraph['qas']:
        output[qa['id']] = {
          'question': qa['question'],
          'context': paragraph['contesxt']
        }
  return output

# def create_dataset(scores):
  

def main():
  if OPTS.scores is not None:
    scores = [json.load(open(p, 'r')) for p in OPTS.scores]
    print(json.dumps({ 'upper_bound': upper_bound(scores)}, indent=2))
  elif OPTS.sweep is not None:
    keep_scores, total_score = sweep(OPTS.sweep)
    keep_scores.sort()
    print(json.dumps({'keep_scores': keep_scores, 'score': total_score}, indent=2))

if __name__ == '__main__':
  OPTS = parse_args()
  main()

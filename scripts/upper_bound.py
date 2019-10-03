# measure f measure for empty answers

import argparse
import json, sys, os

OPTS = None

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--upper-bound', action='store_true', dest='upper_bound',
    help='whether to calculate the upper bound')
  parser.add_argument('--scores', action='store', type=str, nargs="*",
    dest='scores', help="list of scores to create the dataset")
  parser.add_argument('--sweep', type=str, dest='sweep',
    help="directory with results to sweep",)
  parser.add_argument('--construct-dataset', action='store_true',
    dest='construct_dataset', help='whether to construct a dataset')
  parser.add_argument('--original-dataset', default=None, type=str,
    dest='original_dataset', help='the original SQuAD dataset')
  parser.add_argument('--from-sweep', action='store_true',dest='from_sweep',
    help='whether to construct the dataset from the sweeped directories')
  parser.add_argument('--sa-dir', type=str,
    dest='scores_and_answers_dir', help='basedir to get answers and scores from')
  parser.add_argument('--filter-dirs', action='store', type=str, nargs="*",
    dest='filter_dirs', default=None, help='directories to filter from scores_and_answers_dir, '
    'similar to sweeped directories')
  parser.add_argument('--answers', action='store', type=str, nargs="*",
    dest='answers', help='list of answers to create the dataset')
  parser.add_argument('--output-dataset', default=None, type=str,
    dest='output_dataset', help='output file to drop the dataset')

  args = parser.parse_args()
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def make_upper_bound_preds(scores):
  preds = []
  for question_set in zip(*[s.items() for s in scores]):
    # [(id, score), (id, score), ...] -> [(id, id, ...), (score, score, ...)]
    ids, score_set = list(zip(*question_set))
    if len(set(ids)) > 1:
      raise ValueError('Different question ids', ids, score_set)
    correct_index = [idx for idx, value in enumerate(score_set) if value == 1]
    if len(correct_index) > 0:
      preds.append(dict(id=ids[0], correct=correct_index[0]))
  return preds

def calculate_upper_bound(scores):
  upper_bound_preds = make_upper_bound_preds(scores)
  total_correct = len(upper_bound_preds)
  return 100.0 * (total_correct / len(scores[0]))

def get_data_from_dir(data_dir, filename, filter_dirs=None):
  data = {}
  result_dirs = os.listdir(data_dir)
  if filter_dirs is not None and len(filter_dirs) > 0:
    result_dirs = [r for r in result_dirs  if r in filter_dirs]
  for result_dir in result_dirs:
    data_path = os.path.join(data_dir, result_dir, filename)
    if os.path.exists(data_path):
      data[result_dir] = json.load(open(data_path, 'r'))
  return data

def get_answers_from_dir(answers_dir, filter_dirs=None):
  return get_data_from_dir(answers_dir, 'predictions.json', filter_dirs)

def get_scores_from_dir(scores_dir, filter_dirs=None):
  return get_data_from_dir(scores_dir, 'exact_scores.json', filter_dirs)

def sweep(scores_dir):
  max_score = 0.0
  keep_scores = {}
  scores_dict = get_scores_from_dir(scores_dir)
  print('Got scores', scores_dict.keys())
  for score_name, score_value in scores_dict.items():
    score = calculate_upper_bound(list(keep_scores.values()) + [score_value])
    if score > max_score:
      keep_scores[score_name] = score_value
      max_score = score
  return list(keep_scores.keys()), max_score

def squad_to_question_and_context(original_dataset):
  output = {}
  for entry in original_dataset:
    for paragraph in entry['paragraphs']:
      for qa in paragraph['qas']:
        output[qa['id']] = dict(
          question=qa['question'],
          context=paragraph['context']
        )
  return output

def add_answers_to_preds(answers_dict, example_set):
  # set of answers 
  # {system_1: {ans_id: ans}, system_0: {ans_id: ans}...}
  output = []
  for example in example_set:
    answers = [system_values[example['id']] for _, system_values in answers_dict.items()]
    output.append({ **example, 'answers': answers })
  return output

def add_qc_to_preds(question_context_ids, example_set):
  output = []
  for example in example_set:
    example = {**example, **question_context_ids[example['id']] }
    output.append(example)
  return output

def create_dataset(original_dataset, scores_dict, answers_dict):
  question_context_ids = squad_to_question_and_context(original_dataset)
  example_set = make_upper_bound_preds(list(scores_dict.values()))
  example_set = add_answers_to_preds(answers_dict, example_set)
  example_set = add_qc_to_preds(question_context_ids, example_set)

  return example_set

def create_dataset_from_scores_and_answers(scores_dict, answers_dict):
  original_dataset = json.load(open(OPTS.original_dataset, 'r'))['data']
  return create_dataset(original_dataset, scores_dict, answers_dict)

def create_dataset_from_scores_dir(dir, filter_dirs):
  scores_dict = get_scores_from_dir(dir, filter_dirs)
  answers_dict = get_answers_from_dir(dir, filter_dirs)
  return create_dataset_from_scores_and_answers(scores_dict, answers_dict)

def main():
  # You can:
  #   1. calculate the upper bound from score files
  #   2. sweep directories to conform the upper bound
  #   3. create a dataset from score directory

  # calculate upper bound
  if OPTS.upper_bound:
    scores = [json.load(open(p, 'r')) for p in OPTS.scores]
    print(json.dumps({ 'upper_bound': calculate_upper_bound(scores)}, indent=2))

  # calculate sweep
  if OPTS.sweep is not None:
    keep_scores, total_score = sweep(OPTS.sweep)
    keep_scores.sort()
    print(json.dumps({'keep_scores': keep_scores, 'score': total_score}, indent=2))

  # construct dataset
  if OPTS.construct_dataset:
    if OPTS.original_dataset is None:
      raise ValueError('To create a dataset you must provide an SQuAD like dataset')
    if OPTS.from_sweep and OPTS.sweep is None:
      raise ValueError('To create a dataset from sweep you must provide a sweep dir')

    dataset = None
    if OPTS.from_sweep:
      dataset = create_dataset_from_scores_dir(OPTS.sweep, keep_scores)
    else:
      if OPTS.scores_and_answers_dir is not None:
        dataset = create_dataset_from_scores_dir(OPTS.scores_and_answers_dir, OPTS.filter_dirs)
      elif OPTS.scores is not None and OPTS.answers is not None:
        scores = [json.load(open(p, 'r')) for p in OPTS.scores]
        answers = [json.load(open(p, 'r')) for p in OPTS.answers]
        dataset = create_dataset_from_scores_and_answers(scores, answers)
      
    if dataset is not None:
      output = OPTS.output_dataset
      if output is None:
        output = os.path.join(os.getcwd(), 'upper_bound_dataset.json')
      print('Writing dataset to {}'.format(output))
      json.dump(obj=dataset, fp=open(output, 'w'))

if __name__ == '__main__':
  OPTS = parse_args()
  main()

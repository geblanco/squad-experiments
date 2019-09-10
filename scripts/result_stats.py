"""
Get various stats over the results obtained by a model (number of empty,
correct answers etc).
"""

import sys, json, argparse

args = None
dataset = None
predictions = None
scores = None

def parse_args():
  parser = argparse.ArgumentParser('Cross empty answers with exact answer results. Get an idea of how much of the answers were empty.')
  parser.add_argument('--dataset', '-d', help='Input json dataset',
                      default=None, required=True, type=str)
  parser.add_argument('--predictions', '-p', help='Answers from model.',
                      default=None, required=True, type=str)
  parser.add_argument('--scores', '-s', help='Obtained scores from the evaluation script',
                      default=None, required=True, type=str)
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def get_empty_answers_from_dataset(dataset):
  empty_ids = []
  for entry in dataset:
    for paragraph in entry['paragraphs']:
      for qa in paragraph['qas']:
        qas_id = qa['id']
        if qa.get('is_impossible', False) or len(qa['answers']) == 0:
          empty_ids.append(qas_id)
  return empty_ids

def main():
  dataset_empty = get_empty_answers_from_dataset(dataset)
  assert(len(predictions) == len(scores))

  model_empty = [key for key in predictions if predictions[key] == '']
  model_correct = [key for key in scores if scores[key] == 1]
  model_correct_empty = [key for key in model_correct if key in dataset_empty]
  total_nb_questions = len(predictions)

  print('{:21}{}/{} ~ {:.4}'.format('Dataset total empty: ',
    len(dataset_empty), total_nb_questions, len(dataset_empty) / total_nb_questions
  ))
  print('{:21}{}/{} ~ {:.4}'.format('Model total correct: ',
    len(model_correct), total_nb_questions, len(model_correct) / total_nb_questions
  ))
  print('{:21}{}/{} ~ {:.4}'.format('Model total empty: ',
    len(model_empty), total_nb_questions, len(model_empty) / total_nb_questions
  ))
  print('{:21}{}/{} ~ {:.4}'.format('Model correct empty: ',
    len(model_correct_empty), len(model_correct), len(model_correct_empty) / len(model_correct)
  ))

if __name__ == '__main__':
  args = parse_args()
  dataset = json.load(open(args.dataset, 'r'))['data']
  predictions = json.load(open(args.predictions, 'r'))
  scores = json.load(open(args.scores, 'r'))
  main()

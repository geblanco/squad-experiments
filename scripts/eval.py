# measure f measure for empty answers
import math
import argparse
import json, sys

OPTS = None

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('data_file', metavar='data.json', help='Input data JSON file.')
  parser.add_argument('pred_file', metavar='pred.json', help='Model predictions.')
  parser.add_argument('--merge', required=False, type=str, default=None,
    help='Merge metrics with given json file')
  args = parser.parse_args()
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def make_qid_to_has_ans(dataset):
  qid_to_has_ans = {}
  for article in dataset:
    for p in article['paragraphs']:
      for qa in p['qas']:
        qid_to_has_ans[qa['id']] = bool(qa['answers'])
  return qid_to_has_ans

def make_preds_to_has_ans(preds):
  qid_to_has_ans = {}
  for key in preds:
    qid_to_has_ans[key] = bool(preds[key])
  return qid_to_has_ans

def run_precision_recall_analysis(pred_qids, gold_qids, total_questions):
  true_pos = 0
  for qid in gold_qids:
    if qid in pred_qids:
      true_pos += 1

  precision = 0 if len(pred_qids) == 0 else ((1.0 * true_pos) / len(pred_qids))
  recall = 0 if len(gold_qids) == 0 else ((1.0 * true_pos) / len(gold_qids))
  try:
    f = (2 * precision * recall) / (precision + recall)
  except:
    f = 0.0
  return {
    'precision': floor(precision*100),
    'recall': floor(recall*100),
    'f1': floor(f*100),
    'true_pos': true_pos,
    'selected': len(pred_qids),
    'relevant': len(gold_qids),
    'empty_percentage': floor((len(pred_qids) / total_questions) * 100)
  }

def floor(number):
  return math.floor(number * 100)/100.0

def merge(source, dest):
  for key in source.keys():
    dest[key] = source[key]
  return dest

def main():
  dataset = json.load(open(OPTS.data_file, 'r'))['data']
  preds = json.load(open(OPTS.pred_file, 'r'))
  qid_to_has_ans = make_qid_to_has_ans(dataset)
  preds_to_has_ans = make_preds_to_has_ans(preds)

  qid_no_ans = [ key for key in qid_to_has_ans if not qid_to_has_ans[key] ]
  preds_no_ans = [ key for key in preds_to_has_ans if not preds_to_has_ans[key] ]
  eval_dict = run_precision_recall_analysis(preds_no_ans, qid_no_ans, len(qid_to_has_ans))
  if OPTS.merge is not None:
    eval_dict = merge(eval_dict, json.load(open(OPTS.merge, 'r')))
  print(json.dumps(eval_dict, indent=2))

if __name__ == '__main__':
  OPTS = parse_args()
  main()
# measure f measure for empty answers
import math
import argparse
import json, sys

OPTS = None

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('data_file', metavar='data.json', help='Input data JSON file.')
  parser.add_argument('pred_file', metavar='pred.json', help='Model predictions.')
  parser.add_argument('--na-prob-file', help='Model estimates of probability of no answer.')
  parser.add_argument('--na-prob-thresh', type=float, default=1.0,
                      help='Threshold to predict empty. Default to best_f1_thresh from results '
                      '(merge option) or 1.0')
  parser.add_argument('--merge', required=False, type=str, default=None,
                      help='Merge metrics with given json file')
  parser.add_argument('--pretty', action='store_true', help='Whether to pretty print results.')

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

def run_precision_recall_analysis(prefix, pred_qids, gold_qids, total_questions):
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
    prefix + '_precision': floor(precision*100),
    prefix + '_recall': floor(recall*100),
    prefix + '_f1': floor(f*100),
    prefix + '_true_pos': true_pos,
    prefix + '_selected': len(pred_qids),
    prefix + '_relevant': len(gold_qids),
    prefix + '_percentage': floor((len(pred_qids) / total_questions) * 100)
  }

def apply_threshold(preds, na_probs, na_prob_thresh, qid_list):
  new_preds = {}
  for qid in qid_list:
    if na_probs[qid] > na_prob_thresh:
      new_preds[qid] = ''
    else:
      new_preds[qid] = preds[qid]
  return new_preds

def floor(number):
  return math.floor(number * 100)/100.0

def merge(source, dest):
  for key in source.keys():
    dest[key] = source[key]
  return dest

def main():
  dataset = json.load(open(OPTS.data_file, 'r'))['data']
  preds = json.load(open(OPTS.pred_file, 'r'))

  if OPTS.merge is not None:
    orig_results = json.load(open(OPTS.merge, 'r'))
    na_prob_thresh = orig_results['best_f1_thresh']
  else:
    na_prob_thresh = OPTS.na_prob_thresh

  if OPTS.na_prob_file:
    na_probs = json.load(open(OPTS.na_prob_file, 'r'))
  else:
    na_probs = {k: 0.0 for k in preds}
    na_prob_thresh = 1.0

  qid_to_has_ans = make_qid_to_has_ans(dataset)
  new_preds = apply_threshold(preds, na_probs, na_prob_thresh, list(qid_to_has_ans.keys()))
  preds_to_has_ans = make_preds_to_has_ans(new_preds)
  qid_no_ans = [ key for key in qid_to_has_ans if not qid_to_has_ans[key] ]
  preds_no_ans = [ key for key in preds_to_has_ans if not preds_to_has_ans[key] ]
  no_ans_eval_dict = run_precision_recall_analysis('NoAns', preds_no_ans, qid_no_ans, len(qid_to_has_ans))
  qid_has_ans = [ key for key in qid_to_has_ans if qid_to_has_ans[key] ]
  preds_has_ans = [ key for key in preds_to_has_ans if preds_to_has_ans[key] ]
  has_ans_eval_dict = run_precision_recall_analysis('HasAns', preds_has_ans, qid_has_ans, len(qid_to_has_ans))
  eval_dict = merge(has_ans_eval_dict, no_ans_eval_dict)
  if OPTS.merge is not None:
    eval_dict = merge(eval_dict, orig_results)
  if eval_dict.get('NoAns_exact') is not None:
    # For empty answers, exact matches with recall (how many retrieved were relevant)
    eval_dict['NoAns_exact'] = eval_dict['NoAns_recall']
  
  dump_args = {}
  if OPTS.pretty:
    dump_args['indent'] = 2
  
  print(json.dumps(eval_dict, **dump_args))

if __name__ == '__main__':
  OPTS = parse_args()
  main()

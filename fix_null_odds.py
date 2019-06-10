import sys, json, argparse

def get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--preds_file', help="Path to the SQuAD v2 compatible predictions file.")
  parser.add_argument('--null_ods_file', help="Path to the SQuAD v2 compatible null odds file.")
  parser.add_argument('--null_score_value', help="Value to use to fill missing values", default=1.0)
  parser.add_argument('--overwrite', action='store_true', default=True, help="Whether to overwrite source null odds file or generate a new one (default).")
  return parser

def work(predictions_ids, null_odds_ids, output_file):
  predictions_ids = list(predictions.keys())
  null_odds_ids = list(null_odds.keys())

  if len(predictions_ids) == len(null_odds_ids):
    print('No differences!')
    sys.exit(0)

  diff = [d for d in predictions_ids if d not in null_odds_ids]

  for diff_id in diff:
    null_odds[diff_id] = args.null_score_value

  json.dump(fd=open(output_file, 'w'), obj=null_ods, indent=2)

if __name__ == '__main__':
  args, _ = get_parser().parse_known_args()
  
  predictions = json.load(open(args.preds_file, 'r'))
  null_odds = json.load(open(args.null_ods_file, 'r'))

  file = args.null_odds_file
  if not args.overwrite:
    file = args.null_odds_file + '_corrected.json'

  work(predictions_ids, null_odds_ids, file)

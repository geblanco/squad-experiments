import json, argparse

def get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--qa_file', help="Path to the SQuAD v2 compatible json QA file.")
  parser.add_argument('--probs_file', help="Path to the SQuAD v2 compatible probs file.")
  parser.add_argument('--null_ods_file', help="Path to the SQuAD v2 compatible null odds file.")
  parser.add_argument('--ids_file', help="Path to the IDS to work on.")
  parser.add_argument('--isolate', action='store_true', default=False, help="Whether to isolate or avoid (default) the given ids by ids_file.")
  parser.add_argument('--overwrite', action='store_true', default=True, help="Whether to overwrite source file or generate a new one (default).")
  return parser

IDS = None
ISOLATE = False

def keep(id, isolate):
  return (not isolate and id not in IDS) or (isolate and id in IDS)

def filter_qa(paragraph):
  retain = []
  for p in paragraph:
    for qa in p['qas']:
      if keep(qa['id'], ISOLATE):
        retain.append(p)
  return retain

def filter_pa(datapoint):
  filtered = filter_qa(datapoint['paragraphs'])
  if len(filtered):
    return filtered
  return None

def clean_qa(data):
  data = [f for f in list(filter(filter_pa, data)) if f is not None]
  return data

def work_qa(path, overwrite):
  dataset = json.load(open(path, 'r'))
  data = clean_qa(dataset['data'])
  out_path = path
  if overwrite:
    out_path += '_v1.json'
  json.dump(fp=open(out_path, 'w'), obj={'data': data}, ensure_ascii=False, indent=2)

def clean_odds(data):
  data = {d: data[d] for d in data.keys() if keep(d, ISOLATE)}
  return data

def work_odds(path, overwrite):
  odds = json.load(open(path, 'r'))
  odds = clean_odds(odds)
  out_path = path
  if overwrite:
    out_path += '_v1.json'
  json.dump(fp=open(out_path, 'w'), obj=odds, ensure_ascii=False, indent=2)

if __name__ == '__main__':
  args, _ = get_parser().parse_known_args()
  IDS = [line.strip() for line in open(args.ids_file, 'r').readlines()]
  ISOLATE = args.isolate
  # sqa file
  if args.qa_file:
    work_qa(args.qa_file, args.overwrite)

  if args.probs_file:
    work_odds(args.probs_file, args.overwrite)

  if args.null_ods_file:
    work_odds(args.null_ods_file, args.overwrite)
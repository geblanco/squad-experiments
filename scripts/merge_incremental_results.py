import os
import json
import argparse

FLAGS = None

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--results', '-r', help='Results files', action='store',
      nargs='*', required=True)
  return parser.parse_args()

def get_output_path(file_path):
  return os.path.join(os.path.dirname(file_path), 'results.json')

def merge_results(results):
  output = {}
  fields = list(results[0].keys())
  for field in fields:
    try:
      all_data = [r[field] for r in results]
      output[f'{field}_max'] = max(all_data)
      output[f'{field}_min'] = min(all_data)
      output[f'{field}_mean'] = sum(all_data)/len(all_data)
    except Exception as e:
      print('Excepted!', results, fields, FLAGS)
      raise e
  output['nof_results'] = len(results)
  return output

def main():
  results = [json.load(open(f, 'r')) for f in FLAGS.results]
  end_results = merge_results(results)
  output_path = get_output_path(FLAGS.results[0])
  json.dump(fp=open(output_path, 'w'), obj=end_results, indent=2)

if __name__ == '__main__':
  FLAGS = parse_args()
  main()

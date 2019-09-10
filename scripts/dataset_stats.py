from math import sqrt
import sys, json, argparse

args = None
dataset = None
predictions = None
scores = None

def parse_args():
  parser = argparse.ArgumentParser('Get basic stats from a dataset')
  parser.add_argument('--dataset', '-d', help='Input json dataset',
                      default=None, required=True, type=str)
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def mean(group):
  return sum(group)/len(group)

def stddev(group):
  group_mean = mean(group)
  s = [(sub - group_mean) **2 for sub in group]
  return sqrt(sum(s)/(len(group)-1))

def main():
  len_paragraphs = []
  len_questions = []
  len_answers = []
  for entry in dataset:
    for paragraph in entry['paragraphs']:
      len_paragraphs.append(len(paragraph['context']))
      for qa in paragraph['qas']:
        len_questions.append(len(qa['question']))
        if qa['answers']:
          len_answers.extend([len(a['text']) for a in  qa['answers']])
  print('Number of paragraphs {}'.format(len(len_paragraphs)))
  print('Avg paragraph len {:.4} ± {:.4}'.format(mean(len_paragraphs),
    stddev(len_paragraphs)))
  print('Number of questions {}'.format(len(len_questions)))
  print('Avg question len {:.4} ± {:.4}'.format(mean(len_questions),
    stddev(len_questions)))
  print('Avg answer len {:.4} ± {:.4}'.format(mean(len_answers),
    stddev(len_answers)))

if __name__ == '__main__':
  args = parse_args()
  dataset = json.load(open(args.dataset, 'r'))['data']
  main()


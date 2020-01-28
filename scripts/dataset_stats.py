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
  paragraphs = []
  questions = []
  answers = []
  nb_empty_answers = 0
  for entry in dataset:
    for paragraph in entry['paragraphs']:
      paragraphs.append(len(paragraph['context']))
      for qa in paragraph['qas']:
        questions.append(len(qa['question']))
        if qa['answers']:
          answers.extend([len(a['text']) for a in  qa['answers']])
        if qa.get('is_impossible', False) or len(qa['answers']) == 0:
          nb_empty_answers += 1
  len_paragraphs = len(paragraphs)
  mean_par = mean(paragraphs)
  stddev_par = stddev(paragraphs)
  len_questions = len(questions)
  mean_ques = mean(questions)
  stddev_ques = stddev(questions)
  len_answers = len(answers)
  mean_ans = mean(answers)
  stddev_ans = stddev(answers)
  print('Number of paragraphs {}'.format(len_paragraphs))
  print('Avg paragraph len {:.4} ± {:.4}'.format(mean_par, stddev_par))
  print('Number of questions {}'.format(len_questions))
  print('Avg question len {:.4} ± {:.4}'.format(mean_ques, stddev_ques))
  print('Avg answer len {:.4} ± {:.4}'.format(mean_ans, stddev_ans))
  print('Number of empty answer questions {}'.format(nb_empty_answers))
  print('% of empty answer questions {}'.format(nb_empty_answers/len_questions))

if __name__ == '__main__':
  args = parse_args()
  dataset = json.load(open(args.dataset, 'r'))['data']
  main()


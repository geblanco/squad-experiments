#!/usr/bin/env python

"""
Cross answers from different models, this gets an amount of overlap between
different models against the same dataset.
"""
from matplotlib import pyplot as plt
from collections import defaultdict

import json, os, sys
import pandas as pd
import argparse

args = None

def parse_args():
  parser = argparse.ArgumentParser('Cross answers between models over the same dataset.')
  parser.add_argument('--reference', '-r', help='Reference answers to compare',
                      default=None, required=True, type=str)
  parser.add_argument('--answers', '-a', help='Input answers JSON file. ' +
                      'A dictionary with question ids and correct or not flag. ' +
                      'Can be obtained with the modified evaluate SQuAD script.',
                      action='store', default=[], required=True, type=str, nargs='*')
  parser.add_argument('--indices', '-i', help='Indices json file.',
                      default=None, required=True, type=str)
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

plt.rcParams.update({'font.size': 30})
args = None

# def calculate_overlap(answers):
#   n_answers = len(answers)
#   amount = (n_answers * (n_answers-1)) // 2
#   overlap = []
  
#   for i in range(n_answers):
#     overlap.append(len(answers[i]))
#     for j in range(i+1, n_answers):
#       set_a = answers[i]
#       set_b = answers[j]
#       intersection = [key for key in set_a if key in set_b]
#       overlap.append(len(intersection))
#   return overlap

def calculate_overlap(ref, answers):
  overlap = []
  
  for answer_set in answers:
    intersection = [key for key in ref if key in answer_set]
    overlap.append(len(intersection))
  return overlap

def main():
  # get the data
  reference_answers = json.load(open(args.reference, 'r'))
  correct_ref_answers = [a for a in reference_answers if reference_answers[a] == 1]
  
  answers = [json.load(open(a, 'r')) for a in args.answers]
  correct_answers = [[a for a in answer_set if answer_set[a] == 1] for answer_set in answers]
  
  indices = json.load(open(args.indices, 'r'))

  reference_score = len(correct_ref_answers) / len(reference_answers)
  all_answers = [correct_ref_answers]
  for answer_set in correct_answers:
    all_answers.append(answer_set)

  # prepare data as (must be a square matrix):
  #  ref dataset total -     -
  #  
  #  d2 total d2-ref overlap -
  #  d3 total d3-ref overlap d2-d3 overlap
  data = { column: { row: 0 for row in indices['rows'] } for column in indices['columns'] }
  for col_idx, column in enumerate(indices['columns']):
    column_obj = {}
    for row_idx, row in enumerate(indices['rows']):
      if col_idx == 0:
        # diagonal
        column_obj[row] = (len(all_answers[row_idx])/len(reference_answers)) * 100
      elif row_idx == 0:
        # reference row
        column_obj[row] = None
      elif row_idx >= col_idx:
        # calculate behind the diagonal
        column_obj[row] = calculate_overlap(all_answers[row_idx], [all_answers[col_idx-1]])[0]
        column_obj[row] = (column_obj[row]/len(reference_answers)) * 100

    data[column] = column_obj

  print(data)
  df = pd.DataFrame(data)
  # df = df.reindex(index, axis=1)
  # df = df.reindex(sort_index(df.index))
  print(df)
  ax = df.plot(kind='bar', width=0.5)
  ax.set_xticklabels(df.index, rotation='horizontal')
  plt.yticks(list(range(0, 101, 10)))
  plt.grid(axis='y', alpha=0.5)

  # colors = ['#e7a366', '#65ea65']
  # for container in ax.containers[1:]:
  #   for index, rect in enumerate(container.patches[1:]):
  #     rect.set_facecolor(colors[index-1])

  # access the rectangles
  # bc=ax.containers[0]
  # bcr0=bc.patches[0]
  # bcr0.set_edgecolor((0.8, 0.8, 0.7, 1))

  plt.show()

if __name__ == '__main__':
  args = parse_args()
  main()
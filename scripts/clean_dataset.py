import json, argparse

"""
To train Bert, the input questions should only have one answer, mark those
as impossible with more, also, clean up paragraphs with no answer.
"""

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', help='dataset to clean', required=True, type=str)
parser.add_argument('-o', '--output', help='directory to put the clean data', required=True, type=str)
args = parser.parse_args()

dataset = json.load(open(args.dataset, 'r'))
data = dataset['data']

marked_impossible = 0
output_data = []
for entry in data:
  output_entry = {}
  output_paragraphs = []
  for paragraph in entry['paragraphs']:
    output_qas = []
    for qa in paragraph['qas']:
      if (len(qa['answers']) != 1) and (not qa['is_impossible']):
        marked_impossible += 1
        qa['is_impossible'] = True
      output_qas.append(qa)
    paragraph['qas'] = output_qas
    if len(output_qas) > 0:
      output_paragraphs.append(paragraph)
  if len(output_paragraphs) > 0:
    output_entry = entry.copy()
    output_entry['paragraphs'] = output_paragraphs
    output_data.append(output_entry)

print('Initial length', len(data))
print('Final length', len(output_data))
print('Marked as impossible', marked_impossible)
json.dump(fp=open(args.output, 'w'), obj={'data': output_data, 'version': dataset['version']})
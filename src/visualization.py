from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import json
import sys
import os

FLAGS = None

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--embeddings_dir', '-d', type=str, required=True,
      help='dir with question and context embeddings')
  parser.add_argument('--output_dir', type=str, required=True, 
      help='directory to drop the processed data')

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def load_embeddings(embeddings_dir):
  embeddings = []
  embedding_batches = [dir for dir in os.listdir(embeddings_dir) 
      if dir.endswith('.npy')]
  for batch in embedding_batches:
    embedding_batch = np.load(os.path.join(embeddings_dir, batch),
        allow_pickle=True)
    embeddings.extend(embedding_batch)

  return embeddings

def main():
  context_embs = load_embeddings(FLAGS.embeddings_dir, 'context')
  stacked_context_embs = np.vstack(context_embs)

  question_embs = load_embeddings(FLAGS.embeddings_dir, 'question')

if __name__ == '__main__':
  FLAGS = parse_args()
  main()

pca = PCA(n_components=30)
pca.fit_transform(question_embs)
pca.fit_transform(question_embs[0])
question_embs
len(question_embs)
np
q0 = question_embs[0]
q0
q0.shape
np.stack(q0)
np.stack(q0, question_embs[1])
np.stack([q0, question_embs[1]])
np.vstack(q0, question_embs[1])
np.vstack([q0, question_embs[1]])
np.vstack([q0, question_embs[1]]).shape
np.vstack(question_embs)
np.vstack(question_embs).shape
stacked_embs = np.vstack(question_embs).shape
np.hstack(stacked_embs)
np.vstack(stacked_embs)
stacked_embs = np.vstack(question_embs)
np.vstack(stacked_embs)
np.vstack(stacked_embs).shape
np.hstack(stacked_embs).shape
pca.fit(stacked_embs)
transformed_embs = pca.transform(stacked_embs)
transformed_embs.shape
sns.lmplot(data=transformed_embs)
sns.lmplot(x='Score1', y='Score2', data=transformed_embs)
tsne_embs = TSNE().fit_transform(transformed_embs)
tsne_embs.shape
sns.lmplot(x='Score1', y='Score2', data=tsne_embs)
tsne_embs
tsne_embs.shape
f = plt.figure()
ax = plt.subplot()
sc = ax.scatter(tsne_embs[:,0], tsne_embs[:,1])
ax.axis('off')
ax.axis('tight')
plt.show()
pd.DataFrame(tsne_embs)
tsne_df = pd.DataFrame(tsne_embs)
sns.lmplot(x='Score1', y='Score2', data=tsne_df)
tsne_df
sns.lmplot(x='0', y='1', data=tsne_df)
tsne_df.columns
sns.lmplot(x=0, y=1, data=tsne_df)
tsne_df[0]
tsne_df.get_dtype_counts
tsne_df.columns
tsne_df.columns()
tsne_df.head()
for col in tsne_df.columns:
    print(col)
sns.lmplot(x=0, y=1, data=tsne_df)
sns.lmplot(x='0', y=1, data=tsne_df)
tips = sns.load_dataset("tips")
list(tips.columns.values)
tips['total_bill']
tips.head()
tsne_df.columns = ['x', 'y']
sns.lmplot(x='x', y='y', data=tsne_df)
plt.show()
questions
%history



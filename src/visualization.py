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



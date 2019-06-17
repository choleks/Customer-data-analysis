import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.manifold import MDS

df = pd.read_csv('data.csv')

# scale and predict

for column in df.columns:
    df[column] = MinMaxScaler().fit_transform(df[column].values.reshape(-1, 1))

df['cluster'] = OneClassSVM().fit_predict(df)

# use mds to show clusters

embed_df = pd.DataFrame(MDS().fit_transform(df[df.columns[:-1]]), columns=['x0', 'x1'], index=df.index)
embed_df['cluster'] = df['cluster']

for label in sorted(embed_df['cluster'].unique().tolist()):
    df_to_plot = embed_df[embed_df['cluster'] == label]
    plt.scatter(df_to_plot['x0'], df_to_plot['x1'], label=label)

plt.legend()
plt.show()

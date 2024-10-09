
import numpy as np
import pandas as pd
import torch
import umap
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#-------------------------------t-SNE---------------------------
# Concatenate the descriptors
data1 = pd.read_csv('.../ECFP_train.csv')
train_descriptors = data1.iloc[:,1:-1]
print(train_descriptors)
data2 = pd.read_csv('.../ECFP_validate.csv')
validation_descriptors = data2.iloc[:,1:-1]
data3 = pd.read_csv('.../ECFP_external.csv')
external_descriptors = data3.iloc[:,1:-1]
#print(valid_descriptors)

all_descriptors = pd.concat([train_descriptors, test_descriptors, valid_descriptors], axis=0)

# Perform t-SNE dimensionality reductio
embedding = TSNE(n_components=2, perplexity=30, random_state=123).fit_transform(all_descriptors)

# Separate the embeddings of training and validation sets
train_embedding = embedding[:len(train_descriptors)]
validation_embedding = embedding[len(train_descriptors):(len(train_descriptors)+len(validation_descriptors))]
external_embedding = embedding[len(train_descriptors)+len(validation_descriptors):]

# Plot the UMAP embeddings
plt.figure(figsize=(10, 8))
plt.scatter(train_embedding[:, 0], train_embedding[:, 1], color='green', label='Training set')
plt.scatter(test_embedding[:, 0], validation_embedding[:, 1], color='brown', label='Validation set')
plt.scatter(valid_embedding[:, 0], external_embedding[:, 1], color='orange', label='External test Set')
plt.xlabel('TSNE-0')
plt.ylabel('TSNE-1')
plt.legend(fontsize=15)
#plt.xticks([])
#plt.yticks([])
plt.savefig('t-SNE.png',dpi=600)
plt.show()

#--------------------------UMAP-------------------------------
# Read each CSV file and extract descriptors
data_paths = [
    '.../1_8_all.csv',
    '.../2_8_all.csv',
    '.../3_8_all.csv',
    '.../4_8_all.csv',
    '.../5_8_all.csv',
    '.../6_8_all.csv',
    '.../7_8_all.csv',
    '.../8_8_all.csv'
]

all_descriptors = pd.DataFrame()

for path in data_paths:
    data = pd.read_csv(path)
    descriptors = data.iloc[:, 1:-1]
    #print(descriptors)
    all_descriptors = pd.concat([all_descriptors, descriptors], axis=0)

# Perform UMAP dimensionality reduction
#embedding = umap.UMAP(n_neighbors=15,n_components=8,min_dist=0.4,metric='correlation',random_state=90).fit_transform(all_descriptors)
embedding = umap.UMAP(a=None, angular_rp_forest=False, b=None,
     force_approximation_algorithm=False, init='spectral', learning_rate=1.0,
     local_connectivity=1.0, low_memory=False, metric='euclidean',
     metric_kwds=None, min_dist=0.1, n_components=2, n_epochs=None,
     n_neighbors=15, negative_sample_rate=5, output_metric='euclidean',
     output_metric_kwds=None, random_state=42, repulsion_strength=1.0,
     set_op_mix_ratio=1.0, spread=1.0, target_metric='categorical',
     target_metric_kwds=None, target_n_neighbors=-1, target_weight=0.5,
     transform_queue_size=4.0, transform_seed=42, unique=False, verbose=False).fit_transform(all_descriptors)
#embedding = TSNE(n_components=2,perplexity=100, random_state=90).fit_transform(all_descriptors)#perplexity=50, 


# Split embeddings into respective datasets
embeddings = []
start_idx = 0

for path in data_paths:
    data = pd.read_csv(path)
    descriptors = data.iloc[:, 1:-1]
    num_samples = len(descriptors)
    embeddings.append(embedding[start_idx:start_idx + num_samples])
    start_idx += num_samples

# Plot the UMAP embeddings
plt.figure(figsize=(10, 8))

colors = ['green', 'brown', 'orange', 'blue', 'red', 'purple', 'pink', 'gray']  # Define colors for each dataset

for i, embedding_i in enumerate(embeddings):
    plt.scatter(embedding_i[:, 0], embedding_i[:, 1], color=colors[i], label=f'C{i+1}')

plt.legend(fontsize=15)
#plt.xticks([])
#plt.yticks([])
plt.show()


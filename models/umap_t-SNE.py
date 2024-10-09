
import numpy as np
import pandas as pd
import torch
import umap
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Concatenate the descriptors
data1 = pd.read_csv('.../ECFP_train.csv')
train_descriptors = data1.iloc[:,1:-1]
print(train_descriptors)
data2 = pd.read_csv('.../ECFP_validate.csv')
validation_descriptors = data2.iloc[:,1:-1]
data3 = pd.read_csv('.../ECFP_external.csv')
external_descriptors = data3.iloc[:,1:-1]
#print(valid_descriptors)


'''
data1 = pd.read_csv('/home/zhy/machine learning/chongxin/chemical space/cluster-class2/1_8_all.csv')
train_descriptors = data1.iloc[:,1:-1]
print(train_descriptors)
data2 = pd.read_csv('/home/zhy/machine learning/chongxin/chemical space/cluster-class2/2_8_all.csv')
validation_descriptors = data2.iloc[:,1:-1]
data3 = pd.read_csv('/home/zhy/machine learning/chongxin/chemical space/cluster-class2/3_8_all.csv')
external_descriptors = data3.iloc[:,1:-1]
'''

all_descriptors = pd.concat([train_descriptors, test_descriptors, valid_descriptors], axis=0)

# Perform t-SNE dimensionality reductio
embedding = TSNE(n_components=2, perplexity=30, random_state=123).fit_transform(all_descriptors)

# Perform UMAP dimensionality reduction
embedding = umap.UMAP(n_neighbors=14,
                      min_dist=0.5,
                      metric='correlation',
                      random_state=290).fit_transform(all_descriptors)


# Separate the embeddings of training and validation sets
train_embedding = embedding[:len(train_descriptors)]
validation_embedding = embedding[len(train_descriptors):(len(train_descriptors)+len(validation_descriptors))]
external_embedding = embedding[len(train_descriptors)+len(validation_descriptors):]

# Plot the UMAP embeddings
plt.figure(figsize=(10, 8))
plt.scatter(train_embedding[:, 0], train_embedding[:, 1], color='green', label='Training set')
plt.scatter(test_embedding[:, 0], validation_embedding[:, 1], color='brown', label='Validation set')
plt.scatter(valid_embedding[:, 0], external_embedding[:, 1], color='orange', label='External test Set')
#plt.title('UMAP Visualization of Chemical Space')
#plt.xlabel('UMAP Dimension 1')
#plt.ylabel('UMAP Dimension 2')
plt.xlabel('TSNE-0')
plt.ylabel('TSNE-1')
plt.legend(fontsize=15)
#plt.xticks([])
#plt.yticks([])
plt.savefig('t-SNE.png',dpi=600)
plt.show()

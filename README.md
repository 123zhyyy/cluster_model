# cluster_multimodal_models
![Example Image](./TOC.png)
Multimodal modeling based on molecular structure and the relationship between the drug and the disease onset, further explored by disease gene clustering combined with KNN-GCN (K-Nearest Neighbors-Graph Convolutional Network) classifying drugs, was applied to predict drug-induced intrahepatic cholestasis (DIIC) as one main aspect of hepatotoxicity.  

## Descriptions of the files
- feature_preparation_VT: deleting the Mordred descriptors with standard deviations less than 0.10, input file is ‘mordred.csv’.
- feature_preparation_pearson: deleting the Mordred descriptors with Pearson correlation coefficients greater than 0.90.  
- louvain_cluster: identifying gene clusters by louvain algorithm, input file is ‘gene_network.txt’.  
- recursive_feature_elimination: screening the multimodal characteristics with the best 10-fold accuracy, input file is ‘multimodal_characteristics.csv’.  
- machine_learning: SVM, RF, XGB modeling methods, input data are in ‘modeling_data.csv’ file.


## Other links
KNN-GNN is available at [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT).



# cluster_multi-dimensional_modeling_method
![Example Image](./TOC.png)
Multi-dimensional modeling based on molecular structure and the relationship between the drug and the disease onset, further explored by disease gene clustering combined with KNN-GCN (K-Nearest Neighbors-Graph Convolutional Network) classifying drugs, was applied to predict drug-induced intrahepatic cholestasis (DIIC) as one main aspect of hepatotoxicity.  

## Descriptions of the files
- louvain_cluster: identifying gene clusters by louvain algorithm, input file is ‘disease_gene_network.txt’.
- network_proximity: calculating the relative average shortest distance (Zdc) between the drug targets and disease onset clusters, input files are ‘drug_target.txt’ and ‘cluster1-8.txt’ in ‘cluster_genes.zip’.
- preprocess_VT: deleting the Mordred descriptors with standard deviations less than 0.10, input file is ‘mordred.csv’.
- preprocess_pearson: deleting the Mordred descriptors with Pearson correlation coefficients greater than 0.90.  
- process_RFE: screening the multimodal characteristics with the best 10-fold accuracy, input file is ‘multimodal_features.csv’.  
- machine_learning: SVM, RF, XGB modeling methods, input data are sheet A/B/C in ‘modeling_data.xlxs’ file.
- preprocess_data: preprocess the files of compounds and clusters before KNN-GCN.
- construct_graph: KNN constructs the topological network.
- predict_results: GCN predicts the associations between compounds and clusters.
  
## Note:
Network proximity needs to run in a Python 2.7 environment.





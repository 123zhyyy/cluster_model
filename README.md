# cluster_multimodal_models


## explain of the file
1. feature_preparation_VT: deleting the Mordred descriptors with standard deviations less than 0.10, input file is ‘mordred.csv’.  
2. feature_preparation_pearson: deleting the Mordred descriptors with Pearson correlation coefficients greater than 0.90.  
3. louvain_cluster: identifying gene clusters by louvain algorithm, input file is ‘gene_network.txt’.  
4. recursive_feature_elimination: screening the multimodal characteristics with the best 10-fold accuracy, input file is ‘multimodal_characteristics.csv’.  
5. machine_learning: SVM, RF, XGB modeling methods, input data are in ‘modeling_data.csv’ file.

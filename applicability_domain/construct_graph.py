
#-----------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")
import numpy as np
#import os
#import re
import pandas as pd
import scipy.sparse as sp
#import torch as th
#import dgl
#from dgl.data.utils import download, extract_archive, get_download_dir
#from itertools import product
from collections import Counter
from copy import deepcopy
from sklearn.model_selection import KFold
from tqdm import tqdm
import random
random.seed(1234)
np.random.seed(1234)

# -----------------------------------------------------------------------------
def load_data(directory):
    directory = ".../111"
    ID = np.loadtxt(directory + '/drug_similarity.csv', delimiter=",", dtype = float)
    IG = np.loadtxt(directory + '/cluster_similarity.csv', delimiter=",", dtype = float)
    
    ID = pd.DataFrame(ID).reset_index()
    IG = pd.DataFrame(IG).reset_index()
    ID.rename(columns = {'index':'id'}, inplace = True)
    IG.rename(columns = {'index':'id'}, inplace = True)
    ID['id'] = ID['id'] + 1
    IG['id'] = IG['id'] + 1
    
    return ID, IG

# -----------------------------------------------------------------------------
def sample(directory, random_seed):

    all_associations = pd.read_csv(directory + '/pair.csv', names=['gene', 'disease', 'label'])
    known_associations = all_associations.loc[all_associations['label'] == 1]
    unknown_associations = all_associations.loc[all_associations['label'] == 0]
    random_negative = unknown_associations.sample(n=known_associations.shape[0], random_state=random_seed, axis=0)

    sample_df = known_associations.append(random_negative)    
    sample_df.reset_index(drop=True, inplace=True)

    return sample_df

# -----------------------------------------------------------------------------
def obtain_data(directory, isbalance):
    
    ID, IG = load_data(directory)
    
    if isbalance:
        dtp = sample(directory, random_seed = 1234)
    else:
        dtp = pd.read_csv(directory + '/pair.csv', names=['gene', 'disease', 'label'])
        
    gene_ids = list(set(dtp['gene']))
    disease_ids = list(set(dtp['disease']))
    random.shuffle(gene_ids)
    random.shuffle(disease_ids)
    print('# gene = {} | disease = {}'.format(len(gene_ids), len(disease_ids)))

    gene_test_num = int(len(gene_ids) / 5)
    disease_test_num = int(len(disease_ids) / 5)
    print('# Test: gene = {} | disease = {}'.format(gene_test_num, disease_test_num))
    
    knn_x = pd.merge(pd.merge(dtp, ID, left_on = 'disease', right_on = 'id'), IG, left_on = 'gene', right_on = 'id')
    
    label = dtp['label']
    knn_x.drop(labels = ['gene', 'disease', 'label', 'id_x', 'id_y'], axis = 1, inplace = True)
    assert ID.shape[0] + IG.shape[0] == knn_x.shape[1]
    print(knn_x.shape, Counter(label))
    
    return ID, IG, dtp, gene_ids, disease_ids, gene_test_num, disease_test_num, knn_x, label

# -----------------------------------------------------------------------------
def generate_task_Tp_train_test_idx(knn_x):
    kf = KFold(n_splits = 5, shuffle = True, random_state = 1234)

    train_index_all, test_index_all = [], []
    train_id_all, test_id_all = [], []
    fold = 0
    for train_idx, test_idx in tqdm(kf.split(knn_x)): #train_index与test_index为下标
        print('-------Fold ', fold)
        train_index_all.append(train_idx) 
        test_index_all.append(test_idx)

        train_id_all.append(np.array(dtp.iloc[train_idx][['gene', 'disease']]))
        test_id_all.append(np.array(dtp.iloc[test_idx][['gene', 'disease']]))

        print('# Pairs: Train = {} | Test = {}'.format(len(train_idx), len(test_idx)))
        fold += 1
    return train_index_all, test_index_all, train_id_all, test_id_all

# -----------------------------------------------------------------------------
def generate_task_Tb_train_test_idx(item, ids, dtp):
    
    test_num = int(len(ids) / 5)
    
    train_index_all, test_index_all = [], []
    train_id_all, test_id_all = [], []
    
    for fold in range(5):
        print('-------Fold ', fold)
        if fold != 4:
            test_ids = ids[fold * test_num : (fold + 1) * test_num]
        else:
            test_ids = ids[fold * test_num :]

        train_ids = list(set(ids) ^ set(test_ids))
        print('# {}: Train = {} | Test = {}'.format(item, len(train_ids), len(test_ids)))

        test_idx = dtp[dtp[item].isin(test_ids)].index.tolist()
        train_idx = dtp[dtp[item].isin(train_ids)].index.tolist()
        random.shuffle(test_idx)
        random.shuffle(train_idx)
        print('# Pairs: Train = {} | Test = {}'.format(len(train_idx), len(test_idx)))
        assert len(train_idx) + len(test_idx) == len(dtp)

        train_index_all.append(train_idx) 
        test_index_all.append(test_idx)
        
        train_id_all.append(train_ids)
        test_id_all.append(test_ids)
        
    return train_index_all, test_index_all, train_id_all, test_id_all
# -----------------------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report

def generate_knn_graph_save(knn_x, label, n_neigh, train_index_all, test_index_all, pwd, task, balance):
    
    fold = 0
    for train_idx, test_idx in zip(train_index_all, test_index_all): 
        print('-------Fold ', fold)
        
        knn_y = deepcopy(label)
        knn_y[test_idx] = 0
        print('Label: ', Counter(label))
        print('knn_y: ', Counter(knn_y))

        knn = KNeighborsClassifier(n_neighbors = n_neigh)
        knn.fit(knn_x, knn_y)

        knn_y_pred = knn.predict(knn_x)
        knn_y_prob = knn.predict_proba(knn_x)
        knn_neighbors_graph = knn.kneighbors_graph(knn_x, n_neighbors = n_neigh)

        prec_reca_f1_supp_report = classification_report(knn_y, knn_y_pred, target_names = ['label_0', 'label_1'])
        tn, fp, fn, tp = confusion_matrix(knn_y, knn_y_pred).ravel()

        pos_acc = tp / sum(knn_y)
        neg_acc = tn / (len(knn_y_pred) - sum(knn_y_pred)) # [y_true=0 & y_pred=0] / y_pred=0
        accuracy = (tp+tn)/(tn+fp+fn+tp)

        recall = tp / (tp+fn)
        precision = tp / (tp+fp)
        f1 = 2*precision*recall / (precision+recall)

        roc_auc = roc_auc_score(knn_y, knn_y_prob[:, 1])
        prec, reca, _ = precision_recall_curve(knn_y, knn_y_prob[:, 1])
        aupr = auc(reca, prec)

        print('acc={:.4f}|precision={:.4f}|recall={:.4f}|f1={:.4f}|auc={:.4f}|aupr={:.4f}|pos_acc={:.4f}|neg_acc={:.4f}'.format(accuracy, precision, recall, f1, roc_auc, aupr, pos_acc, neg_acc))
        print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
        print('y_pred: ', Counter(knn_y_pred))
        print('y_true: ', Counter(knn_y))
        print('knn_score = {:.4f}'.format(knn.score(knn_x, knn_y)))

        sp.save_npz(pwd + 'task_' + task + '_' + balance + '_testlabel0_knn' + str(n_neigh) + 'neighbors_edge_fold' + str(fold) + '.npz', knn_neighbors_graph)
        fold += 1
    return knn_x, knn_y, knn, knn_neighbors_graph


# Run--------------------------------------------------------------------

for isbalance in [True]:
    print('************isbalance = ', isbalance)
    
    for task in ['Tp', 'Tb']:
        print('=================task = ', task)
        
        ID, IG, dtp, gene_ids, disease_ids, gene_test_num, disease_test_num, knn_x, label = obtain_data(".../111", isbalance)

        if task == 'Tp':
            train_index_all, test_index_all, train_id_all, test_id_all = generate_task_Tp_train_test_idx(knn_x)
        elif task == 'Tb':
            item = 'gene'
            ids = gene_ids
            train_index_all, test_index_all, train_id_all, test_id_all = generate_task_Tb_train_test_idx(item, ids, dtp)
 
        balance = 'balance'

        np.savez_compressed('.../111/construct_graph/task_' + task + '_' + balance + '_testlabel0_knn_edge_train_test_index_all.npz', 
                               train_index_all = train_index_all, 
                               test_index_all = test_index_all,
                               train_id_all = train_id_all, 
                               test_id_all = test_id_all)
        
        pwd = '.../construct_graph/knn/'
        for n_neigh in [1,3,5,7,9,11,13,15]: 
            print('--------------------------n_neighbors = ', n_neigh)
            knn_x, knn_y, knn, knn_neighbors_graph = generate_knn_graph_save(knn_x, label, n_neigh, train_index_all, test_index_all, pwd, task, balance)


# -----------------------------------------------------------------------------
node_feature_label = pd.concat([dtp, knn_x], axis = 1)
pwd = '.../construct_graph/'
node_feature_label.to_csv(pwd + 'node_feature_label_balance.csv')

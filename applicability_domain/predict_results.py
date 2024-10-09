
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from collections import Counter
import re
from tqdm import trange

pwd = '.../'
disease_id_name = pd.read_csv(pwd + 'drug_name.csv')
gene_id_name = pd.read_csv(pwd + 'cluster_name.csv')

# -----------------------------------------------------------------------------

def metrics(y_true, y_pred, y_prob):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    pos_acc = tp / sum(y_true)
    neg_acc = tn / (len(y_pred) - sum(y_pred)) # [y_true=0 & y_pred=0] / y_pred=0
    accuracy = (tp+tn)/(tn+fp+fn+tp)
    
    recall = tp / (tp+fn)
    precision = tp / (tp+fp)
    f1 = 2*precision*recall / (precision+recall)
    
    roc_auc = roc_auc_score(y_true, y_prob)
    prec, reca, _ = precision_recall_curve(y_true, y_prob)
    aupr = auc(reca, prec)
    average1 = (accuracy + precision + recall + roc_auc + aupr) / 5
    average2 = (accuracy + f1 + roc_auc + aupr) / 4
    average3 = (f1 + aupr) / 2
    print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
    print('y_pred: 0 = {} | 1 = {}'.format(Counter(y_pred)[0], Counter(y_pred)[1]))
    print('y_true: 0 = {} | 1 = {}'.format(Counter(y_true)[0], Counter(y_true)[1]))
    print('acc={:.4f}|precision={:.4f}|recall={:.4f}|f1={:.4f}|auc={:.4f}|aupr={:.4f}|pos_acc={:.4f}|neg_acc={:.4f}'.format(accuracy, precision, recall, f1, roc_auc, aupr, pos_acc, neg_acc))
    print('{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(accuracy, precision, recall, f1, roc_auc, aupr, average1, average2, average3))
    
def train_test_file(task, balance):
    
    train_test_id_idx = np.load('/home/jxy/111/construct_graph/task_' + task +'_' + balance + '_testlabel0_knn_edge_train_test_index_all.npz', allow_pickle = True)
    train_index_all = train_test_id_idx['train_index_all']
    test_index_all = train_test_id_idx['test_index_all']
    train_id_all = train_test_id_idx['train_id_all'] # 'gene', 'disease'
    test_id_all = train_test_id_idx['test_id_all'] # 'gene', 'disease'
    return test_index_all, test_id_all, (train_index_all, train_id_all)

def balanced_results_file(task, knn, lr, fold): 
    file = np.load('.../111/train_results/' + 'task_' + task + '_balance_' + knn + '_lr' + str(lr) + '_fold' + str(fold) + '.npz')
    y_true_train, y_pred_train, y_prob_train = file['ys_train'][0], file['ys_train'][1], file['ys_train'][2]
    y_true_test, y_pred_test, y_prob_test = file['ys_test'][0], file['ys_test'][1], file['ys_test'][2] 
    
    print('Train:')
    metrics(y_true_train, y_pred_train, y_prob_train)
    print('Test:')
    metrics(y_true_test, y_pred_test, y_prob_test)
    
    return y_true_test, y_pred_test, y_prob_test, (y_true_train, y_pred_train, y_prob_train)


def sample(random_seed):
    all_associations = pd.read_csv('.../111/pair.csv', names=['gene', 'disease', 'label'])
    known_associations = all_associations.loc[all_associations['label'] == 1]
    unknown_associations = all_associations.loc[all_associations['label'] == 0]
    random_negative = unknown_associations.sample(n=known_associations.shape[0], random_state=random_seed, axis=0)

    sample_df = known_associations.append(random_negative)
    sample_df.reset_index(drop=True, inplace=True)

    return sample_df


def run_balanced_Tp(task, balance, knn, lr):
    test_index_all, test_id_all, _ = train_test_file(task, balance)

    for i in range(5):
        print('==== Fold ', i)
        y_true_test, y_pred_test, y_prob_test, _ = balanced_results_file(task, knn, lr, fold = i)
        
        results_df = pd.DataFrame(test_id_all[i].reshape(-1,2))
        if i == 0:
             y_true_test_all, y_pred_test_all, y_prob_test_all = y_true_test, y_pred_test, y_prob_test
             results_df_all = results_df
        else:
            y_true_test_all = np.hstack([y_true_test_all, y_true_test])
            y_pred_test_all = np.hstack([y_pred_test_all, y_pred_test])
            y_prob_test_all = np.hstack([y_prob_test_all, y_prob_test])
            results_df_all = np.vstack([results_df_all, results_df])
            
    results_df_all = pd.DataFrame(results_df_all, columns = ['gene', 'disease'])
    results_df_all['y_true'] = y_true_test_all
    results_df_all['y_pred'] = y_pred_test_all
    results_df_all['y_prob'] = y_prob_test_all
    
    results_df_all = pd.merge(results_df_all, gene_id_name, left_on = 'gene', right_on = 'id_gene')
    results_df_all = pd.merge(results_df_all, disease_id_name, left_on = 'disease', right_on = 'id_disease')
    results_df_all.drop(labels = ['id_gene', 'id_disease'], axis = 1, inplace = True)
    results_df_all.sort_values(by = ['disease_y', 'y_prob'], ascending = False, inplace = True)
    
    pwd = ".../predict_results/"
    results_df_all.to_csv(pwd + task + '_balance_predict_results.csv')
    
    return results_df_all

def run_balanced_Tb(task, balance, knn, lr):
    dtp = sample(random_seed = 1234)
    test_index_all, test_id_all, _ = train_test_file(task, balance)

    for i in range(5):
        print('==== Fold ', i)
        y_true_test, y_pred_test, y_prob_test, _ = balanced_results_file(task, knn, lr, fold = i)

        temp = dtp.iloc[test_index_all[i]][['gene', 'disease']]
        if i == 0:
            y_true_test_all, y_pred_test_all, y_prob_test_all = y_true_test, y_pred_test, y_prob_test
            
            results_df = temp
        else:
            y_true_test_all = np.hstack([y_true_test_all, y_true_test])
            y_pred_test_all = np.hstack([y_pred_test_all, y_pred_test])
            y_prob_test_all = np.hstack([y_prob_test_all, y_prob_test])
            
            results_df = pd.concat([results_df, temp], axis = 0)
            
    results_df['y_true'] = y_true_test_all.reshape(-1)
    results_df['y_pred'] = y_pred_test_all.reshape(-1)
    results_df['y_prob'] = y_prob_test_all.reshape(-1)

    results_df = pd.merge(results_df, gene_id_name, left_on = 'gene', right_on = 'id_gene')
    results_df = pd.merge(results_df, disease_id_name, left_on = 'disease', right_on = 'id_disease')
    results_df.drop(labels = ['id_gene', 'id_disease'], axis = 1, inplace = True)
    results_df.sort_values(by = ['disease_y', 'y_prob'], ascending = False, inplace = True)
    
    pwd = ".../predict_results/"
    results_df.to_csv(pwd + task + '_balance_precdict_results.csv')
    
    return results_df

# -----------------------------------------------------------------------------

results_Tp_balanced = run_balanced_Tp(task = 'Tp', balance = 'balance', knn = '9knn', lr = 0.001)

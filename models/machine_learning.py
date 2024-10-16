
import shap
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import sklearn.metrics  
from sklearn.metrics import make_scorer,classification_report,accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef,confusion_matrix,roc_auc_score, roc_curve,auc



#——————————————————————————————————————————————————————
data = pd.read_csv('///.csv')
X = data.iloc[:,1:-1]
#print(X)
y = data['Class']
#print(y)

data2 = pd.read_csv('///test.csv')
X_test = data2.iloc[:,1:-1]
#print(X)
y_test = data2['Class']
#print(y)

i=1234
acc=[]
pre=[]
se=[]
sp=[]
f1=[]
mcc=[]
auc=[]

acc2=[]
pre2=[]
se2=[]
sp2=[]
f12=[]
mcc2=[]
auc2=[]

acc3=[]
pre3=[]
se3=[]
sp3=[]
f13=[]
mcc3=[]
auc3=[]

#——————————————————————————————————————————————————————
for random_state in range(100): 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True,random_state=1244)
    print(X_train.shape, y_train.shape)
    print(sum(y_train))
    print(X_validation.shape, y_validation.shape)
    print(sum(y_validation))
    print(X_test.shape, y_test.shape)
    print(sum(y_test))
    i += 10
'''
clf = svm.SVC(random_state=90)
param_grid=[
    {'kernel':['rbf'],'C':[1,10,100,1000],'gamma':[0.0001,0.001,0.01,0.1]},
    {'kernel':['linear'],'C':[1,10,100,1000]}
    ]

clf = XGBClassifier(random_state=90)
param_grid={
    'n_estimators':np.arange(10,201,10),
    'max_depth':np.arange(2,11,1),
    'learning_rate':np.arange(0.1,0.3,0.1),
    'gamma':np.arange(0,0.5,0.1)
    }

clf = RandomForestClassifier(random_state=90)
param_grid={
    'n_estimators':np.arange(10,501,10),
    'max_depth':np.arange(1,11,1),
    }

grid=GridSearchCV(clf,param_grid,cv=10,scoring='accuracy',verbose=1,n_jobs=-1)
grid.fit(X_train,y_train)
print(grid.best_params_,
      grid.best_score_,
      grid.best_estimator_,
      grid.best_index_,)
'''
    clf = svm.SVC(kernel='rbf',C=100,gamma=0.0001,probability=True, random_state=90)
    #clf = XGBClassifier(n_estimators=20, max_depth=8, learning_rate=0.1, gamma=0.2, random_state=90)
    #clf = RandomForestClassifier(n_estimators=80, max_depth=7, random_state=90)
    score1 = cross_val_score(clf,X_train,y_train,cv=10,scoring=make_scorer(accuracy_score)).mean()
    print("accuracy",score)
    score2 = cross_val_score(clf,X_train,y_train,cv=10,scoring=make_scorer(precision_score)).mean()
    print("precision",score)
    score3 = cross_val_score(clf,X_train,y_train,cv=10,scoring=make_scorer(recall_score)).mean()
    print("recall",score)
    predicted=cross_val_predict(clf,X_train,y_train,cv=10)
    conf_matrix= confusion_matrix(y_train,predicted)
    TN=conf_matrix[0,0]
    FP=conf_matrix[0,1]
    specificity=TN/(TN+FP)
    print('specificity',specificity)
    score4 = cross_val_score(clf,X_train,y_train,cv=10,scoring=make_scorer(f1_score)).mean()
    print("f1",score)
    score5 = cross_val_score(clf,X_train,y_train,cv=10,scoring=make_scorer(matthews_corrcoef)).mean()
    print("matthews_corrcoef",score)
    score6 = cross_val_score(clf,X_train,y_train,cv=10,scoring=make_scorer(roc_auc_score)).mean()
    print("AUC",score)

    acc.append(score1)
    pre.append(score2)
    se.append(score3)
    sp.append(specificity)
    f1.append(score4)
    mcc.append(score5)
    auc.append(score6)

#——————————————————————————————————————————————————————
    clf.fit(X_train,y_train)
    y_pred2 = clf.predict(X_validation)
    y_prob2 = clf.predict_proba(X_validation)

    tn2,fp2,fn2,tp2=confusion_matrix(y_validation,y_pred2).ravel()

    score12=accuracy_score(y_validation, y_pred2)
    score22=balanced_accuracy_score(y_validation, y_pred2)
    score32=precision_score(y_validation, y_pred2, average='macro')
    score42=recall_score(y_validation, y_pred2, average='macro')
    specificity2=tn2/(tn2+fp2)
    score52=f1_score(y_validation, y_pred2, average='macro')
    score62=matthews_corrcoef(y_validation, y_pred2)#matthews_corrcoef
    score72 = roc_auc_score(y_validation, y_prob2[:, 1])

    acc2.append(score12)
    pre2.append(score32)
    se2.append(score42)
    sp2.append(specificity2)
    f12.append(score52)
    mcc2.append(score62)
    auc2.append(score72)

#————————————————————————————————————————————————————————
    clf.fit(X_train,y_train)
    y_pred3 = clf.predict(X_external)
    #print(y_pred)
    y_prob3 = clf.predict_proba(X_external)

    tn,fp,fn,tp=confusion_matrix(y_external,y_pred3).ravel()

    score13=accuracy_score(y_external,y_pred3)
    score23=balanced_accuracy_score(y_external, y_pred3)
    score33=precision_score(y_external, y_pred3, average='weighted')
    score43=recall_score(y_external, y_pred3, average='weighted')
    specificity3=tn/(tn+fp)
    score53=f1_score(y_external, y_pred3, average='weighted')
    score63=matthews_corrcoef(y_external, y_pred3)#matthews_corrcoef
    score73 = roc_auc_score(y_external, y_prob3[:, 1])#得到ROC曲线下面积auc,使用每个样本标签“1”的预测概率，得到的是rocauc_1

    acc3.append(score23)
    pre3.append(score33)
    se3.append(score43)
    sp3.append(specificity3)
    f13.append(score53)
    mcc3.append(score63)
    auc3.append(score73)
    print(i-1234)

'''
#————————————————————————————————————————————————————————
clf.fit(X_train,y_train)
column = data.columns[1:-1]
feature_names = list(column)
print(feature_names)
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_train)
print(shap_values)
shap_df = pd.DataFrame(shap_values, columns=feature_names)
shap_df.to_csv('///.csv', index=False)
shap.summary_plot(shap_values, X_train, feature_names=feature_names, plot_type="bar", max_display=8)
shap.summary_plot(shap_values, X_train, feature_names=feature_names, max_display=8)
'''

metrics_df1 = pd.DataFrame({
    'Accuracy': acc,
    'Precision': pre,
    'Recall': se,
    'Specificity': sp,
    'F1-Score': f1,
    'MCC': mcc,
    'AUC': auc,
    'Accuracy2': acc2,
    'Precision2': pre2,
    'Recall2': se2,
    'Specificity2': sp2,
    'F1-Score2': f12,
    'MCC2': mcc2,
    'AUC2': auc2,
    'Accuracy3': acc3,
    'Precision3': pre3,
    'Recall3': se3,
    'Specificity3': sp3,
    'F1-Score3': f13,
    'MCC3': mcc3,
    'AUC3': auc3
})

metrics_df1.to_csv('.../all 100 times.csv', index=False)
print("finish 1")

accuracy_mean = np.mean(acc)
accuracy_std = np.std(acc)
precision_mean = np.mean(pre)
precision_std = np.std(pre)
recall_mean = np.mean(se)
recall_std = np.std(se)
specificity_mean = np.mean(sp)
specificity_std = np.std(sp)
f1_score_mean = np.mean(f1)
f1_score_std = np.std(f1)
mcc_mean = np.mean(mcc)
mcc_std = np.std(mcc)
auc_mean = np.mean(auc)
auc_std = np.std(auc)

accuracy2_mean = np.mean(acc2)
accuracy2_std = np.std(acc2)
precision2_mean = np.mean(pre2)
precision2_std = np.std(pre2)
recall2_mean = np.mean(se2)
recall2_std = np.std(se2)
specificity2_mean = np.mean(sp2)
specificity2_std = np.std(sp2)
f1_score2_mean = np.mean(f12)
f1_score2_std = np.std(f12)
mcc2_mean = np.mean(mcc2)
mcc2_std = np.std(mcc2)
auc2_mean = np.mean(auc2)
auc2_std = np.std(auc2)

accuracy3_mean = np.mean(acc3)
accuracy3_std = np.std(acc3)
precision3_mean = np.mean(pre3)
precision3_std = np.std(pre3)
recall3_mean = np.mean(se3)
recall3_std = np.std(se3)
specificity3_mean = np.mean(sp3)
specificity3_std = np.std(sp3)
f1_score3_mean = np.mean(f13)
f1_score3_std = np.std(f13)
mcc3_mean = np.mean(mcc3)
mcc3_std = np.std(mcc3)
auc3_mean = np.mean(auc3)
auc3_std = np.std(auc3)

metrics_df2 = pd.DataFrame({
    'Acuracy': f"{accuracy_mean:.3f}±{accuracy_std:.3f}",
    'Precision': f"{precision_mean:.3f}±{precision_std:.3f}",
    'Recall': f"{recall_mean:.3f}±{recall_std:.3f}",
    'Specificity': f"{specificity_mean:.3f}±{specificity_std:.3f}",
    'F1-Score': f"{f1_score_mean:.3f}±{f1_score_std:.3f}",
    'MCC': f"{mcc_mean:.3f}±{mcc_std:.3f}",
    'AUC': f"{auc_mean:.3f}±{auc_std:.3f}",
    'Accuracy2': f"{accuracy2_mean:.3f}±{accuracy2_std:.3f}",
    'Precision2': f"{precision2_mean:.3f}±{precision2_std:.3f}",
    'Recall2': f"{recall2_mean:.3f}±{recall2_std:.3f}",
    'Specificity2': f"{specificity2_mean:.3f}±{specificity2_std:.3f}",
    'F1-Score2': f"{f1_score2_mean:.3f}±{f1_score2_std:.3f}",
    'MCC2': f"{mcc2_mean:.3f}±{mcc2_std:.3f}",
    'AUC2': f"{auc2_mean:.3f}±{auc2_std:.3f}",
    'Accuracy3': f"{accuracy3_mean:.3f}±{accuracy3_std:.3f}",
    'Precision3': f"{precision3_mean:.3f}±{precision3_std:.3f}",
    'Recall3': f"{recall3_mean:.3f}±{recall3_std:.3f}",
    'Specificity3': f"{specificity3_mean:.3f}±{specificity3_std:.3f}",
    'F1-Score3': f"{f1_score3_mean:.3f}±{f1_score3_std:.3f}",
    'MCC3': f"{mcc3_mean:.3f}±{mcc3_std:.3f}",
    'AUC3': f"{auc3_mean:.3f}±{auc3_std:.3f}"
}, index=[0])

metrics_df2.to_csv('.../calculation.csv', index=False)
print("finish 2")

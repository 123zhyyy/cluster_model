
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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True,random_state=1244)
#X_train, X_sum, y_train, y_sum = train_test_split(X, y, test_size=0.4, shuffle=True,random_state=2)
#X_test, X_validation, y_test, y_validation = train_test_split(X_sum, y_sum, test_size=0.5, shuffle=True,random_state=2)
print(X_train.shape, y_train.shape)#
print(sum(y_train))
print(X_test.shape, y_test.shape)#
print(sum(y_test))
#print(X_validation.shape, y_validation.shape)#
#print(sum(y_validation))


#——————————————————————————————————————————————————————
clf = svm.SVC(random_state=90)
param_grid=[
    {'kernel':['rbf'],'C':[1,10,100,1000],'gamma':[0.0001,0.001,0.01,0.1]},
    {'kernel':['linear'],'C':[1,10,100,1000]}
    ]
'''
clf = XGBClassifier(random_state=90)
param_grid={
    'n_estimators':np.arange(10,201,10),
    'max_depth':np.arange(2,11,1),
    'learning_rate':np.arange(0.1,0.3,0.1),
    'gamma':np.arange(0,0.5,0.1)
    }
'''
'''
clf = RandomForestClassifier(random_state=90)
param_grid={
    'n_estimators':np.arange(10,501,10),
    'max_depth':np.arange(1,11,1),
    }
'''
grid=GridSearchCV(clf,param_grid,cv=10,scoring='accuracy',verbose=1,n_jobs=-1)
grid.fit(X_train,y_train)
print(grid.best_params_,
      grid.best_score_,
      grid.best_estimator_,
      grid.best_index_,)



#——————————————————————————————————————————————————————
clf = svm.SVC(kernel='rbf',C=100,gamma=0.0001,probability=True, random_state=90)
#clf = XGBClassifier(n_estimators=20, max_depth=8, learning_rate=0.1, gamma=0.2, random_state=90)
#clf = RandomForestClassifier(n_estimators=80, max_depth=7, random_state=90)
score = cross_val_score(clf,X_train,y_train,cv=10,scoring=make_scorer(accuracy_score)).mean()#十折交叉验证的准确度
print("accuracy",score)
score = cross_val_score(clf,X_train,y_train,cv=10,scoring=make_scorer(precision_score)).mean()#十折交叉验证的精密度
print("precision",score)
score = cross_val_score(clf,X_train,y_train,cv=10,scoring=make_scorer(recall_score)).mean()#十折交叉验证的召回率
print("recall",score)
predicted=cross_val_predict(clf,X_train,y_train,cv=10)
conf_matrix= confusion_matrix(y_train,predicted)
TN=conf_matrix[0,0]
FP=conf_matrix[0,1]
specificity=TN/(TN+FP)
print('specificity',specificity)
score = cross_val_score(clf,X_train,y_train,cv=10,scoring=make_scorer(f1_score)).mean()#十折交叉验证的准确度f1-score
print("f1",score)
score = cross_val_score(clf,X_train,y_train,cv=10,scoring=make_scorer(matthews_corrcoef)).mean()#十折交叉验证的准确度f1-score
print("matthews_corrcoef",score)
score = cross_val_score(clf,X_train,y_train,cv=10,scoring=make_scorer(roc_auc_score)).mean()#十折交叉验证的auc
print("AUC",score)


#——————————————————————————————————————————————————————
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)

tn,fp,fn,tp=confusion_matrix(y_test,y_pred).ravel()
print('tn,fp,fn,tp',tn,fp,fn,tp)
specificity=tn/(tn+fp)


print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print('specificity',specificity)
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
print("MCC:", matthews_corrcoef(y_test, y_pred))#matthews_corrcoef
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
cm = pd.DataFrame(cm, index=['0', '1'], columns=['0', '1'])
ax = sns.heatmap(cm, annot=True, cmap='Blues', cbar=False)
ax.set_xlabel('True Label')
ax.set_ylabel('Predicted Label')
plt.show()



roc_auc = roc_auc_score(y_test, y_prob[:, 1])
print('AUC',roc_auc)
fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
#print(fpr,tpr,thresholds)
plt.plot(fpr,tpr)
plt.plot([0, 1], [0, 1], linestyle='--', color='k')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC curve")
plt.show()
print('AUC',auc(fpr, tpr))


#————————————————————————————————————————————————————————
data2 = pd.read_csv('///external.csv')
X_external = data2.iloc[:,1:-1]
#print(X)
y_external = data2['Class']
#print(y)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_external)
print(y_pred)
y_prob = clf.predict_proba(X_external)

tn,fp,fn,tp=confusion_matrix(y_external,y_pred).ravel()
print('tn,fp,fn,tp',tn,fp,fn,tp)
specificity=tn/(tn+fp)

print(classification_report(y_external, y_pred))
print("Accuracy:", accuracy_score(y_external, y_pred))
print("Balanced Accuracy:", balanced_accuracy_score(y_external, y_pred))
print("Precision:", precision_score(y_external, y_pred, average='weighted'))
print("Recall:", recall_score(y_external, y_pred, average='weighted'))
print('specificity',specificity)
print("F1 Score:", f1_score(y_external, y_pred, average='weighted'))
print("MCC:", matthews_corrcoef(y_external, y_pred))#matthews_corrcoef
print("Confusion Matrix:")
print(confusion_matrix(y_external, y_pred))

cm = confusion_matrix(y_external, y_pred)
cm = pd.DataFrame(cm, index=['0', '1'], columns=['0', '1'])
ax = sns.heatmap(cm, annot=True, cmap='Blues', cbar=False)
ax.set_xlabel('True Label')
ax.set_ylabel('Predicted Label')
plt.show()

roc_auc = roc_auc_score(y_external, y_prob[:, 1])#得到ROC曲线下面积auc,使用每个样本标签“1”的预测概率，得到的是rocauc_1
print('AUC',roc_auc)
fpr, tpr, thresholds = roc_curve(y_external, y_prob[:, 1])
#print('绘制PR曲线的数据',fpr,tpr,thresholds)
plt.plot(fpr,tpr)#绘制ROC曲线
plt.plot([0, 1], [0, 1], linestyle='--', color='k')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC curve")
plt.show()
print('AUC',auc(fpr, tpr))

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


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

data = pd.read_csv('///multimodal_characteristics.csv')
X = data.iloc[:,1:-1]
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True,random_state=90)
print(X_train.shape, y_train.shape)
print(sum(y_train))
print(X_test.shape, y_test.shape)
print(sum(y_test))

clf = RandomForestClassifier(random_state=55)
score = []
for i in range(1, 16, 1):
    X_wrapper = RFE(clf, n_features_to_select=i, step=1).fit_transform(X_train, y)
    once = cross_val_score(clf, X_wrapper, y, cv=10,scoring='accuracy').mean()
    score.append(once)
print(max(score), (score.index(max(score))*1)+1)
print(score)
plt.figure(figsize=[10, 5])
plt.title('5-fold cv of RFE')
plt.xlabel('the number of features')
plt.ylabel('accuracy')
plt.plot(range(1, 16, 1), score)
plt.xticks(range(1,16, 1))
plt.show()
print(max(score), (score.index(max(score))*1)+1)print(score)

'''
#RFE
clf = RandomForestClassifier(random_state=90)
selector = RFE(clf, n_features_to_select=8, step=1).fit(X_train, y)

print(selector.support_)
#print(selector.ranking_)
print(selector.n_features_)
X_wrapper = selector.transform(X)
score =cross_val_score(clf, X_wrapper, y, cv=10,scoring='accuracy').mean()print(score)

# save to txt file
np.set_printoptions(threshold=np.inf)
with open("/home/zhy/machine learning/cholestasis Zdc/shujvyizhi/chongxin/8mordred.txt",'w')as f:
    print(selector.support_,file=f)
#    print(select_name0,file=f)
'''


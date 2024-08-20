
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

data = pd.read_csv('///.csv')

X = data.iloc[:,1:-1]
print(X)
y = data['Class']
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True,random_state=90)

sel = VarianceThreshold(threshold=0.0001)
X_var = sel.fit_transform(X)
print(X_var.shape)

file_path = r"///output.csv"
X_var_y = pd.concat((pd.DataFrame(X_var),y_train),axis=1)
X_var_y.to_csv(file_path,index=False)

all_name = X.columns.values.tolist()
select_name_index0 = sel.get_support(indices=True) 
select_name0 = []
for i in select_name_index0:
    select_name0.append(all_name[i])
print(select_name0)

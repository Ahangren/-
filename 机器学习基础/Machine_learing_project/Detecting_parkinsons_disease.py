
"""
未优化版本
"""
# import numpy as np
# import pandas as pd
# import os,sys
#
# from numexpr import nthreads
# from pygments.lexers.objective import objective
# from sklearn.preprocessing import MinMaxScaler
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
#
# df=pd.read_csv('./data/parkinsons.data',sep=',')
# # df2=pd.read_csv('./data/parkinsons.names',sep=',')
# print(df.head())
# # print(df2.head())
# df=df.fillna(df.mean())
# features=df.loc[:,df.columns!='status'].values[:,1:]
# labels=df.loc[:,'status'].values
#
# print(labels[labels==1].shape[0],labels[labels==0].shape[0])
#
# scaler=MinMaxScaler((-1,1))
# x=scaler.fit_transform(features)
# y=labels
#
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=39)
#
# model=XGBClassifier(
#     base_score=0.5,
#     booster='gbtree',
#     colsample_bylevel=1,
#     learning_rate=0.1,
#     gamma=0,
#     max_delta_step=0,
#     max_depth=3,
#     min_child_weight=1,
#     missing=None,
#     n_estimators=100,
#     n_jobs=1,
#     nthread=None,
#     objective='binary:logistic',
#     random_state=0,
#     reg_alpha=0,
#     reg_lambda=1,
#     scale_pos_weight=1,
#     seed=None,
#     silent=None,
#     subsample=1,
#     verbosity=1
# )
# model.fit(x_train,y_train)
#
# y_pred=model.predict(x_test)
# print(accuracy_score(y_test,y_pred)*100)


"""
优化版本
"""
import numpy as np
import pandas as pd
import os,sys

from numexpr import nthreads
from pygments.lexers.objective import objective
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df=pd.read_csv('./data/parkinsons.data',sep=',')
# df2=pd.read_csv('./data/parkinsons.names',sep=',')
print(df.head())
# print(df2.head())
df=df.fillna(df.mean())
features=df.loc[:,df.columns!='status'].values[:,1:]
labels=df.loc[:,'status'].values

print(labels[labels==1].shape[0],labels[labels==0].shape[0])

scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=labels

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=39)

# 使用网格搜索优化参数（参考[1][4][6][8]）
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],  # 控制步长，防止过拟合[4,8](@ref)
    'max_depth': [3, 5, 7],             # 树深度，平衡复杂度与过拟合[4,6](@ref)
    'min_child_weight': [1, 3, 5],       # 叶子节点最小样本权重[6,7](@ref)
    'subsample': [0.7, 0.9],            # 样本随机采样比例[6,8](@ref)
    'colsample_bytree': [0.7, 0.9],     # 特征随机采样比例[6,8](@ref)
    'n_estimators': [100, 200],         # 树的数量[1,4](@ref)
    'reg_alpha': [0, 0.1],              # L1正则化[4,7](@ref)
    'reg_lambda': [1, 0.1]              # L2正则化[4,7](@ref)
}
from sklearn.model_selection import GridSearchCV
grid_search=GridSearchCV(
    estimator=XGBClassifier(objective='binary:logistic',random_state=39),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(x_train,y_train)
best_model=grid_search.best_estimator_
# model=XGBClassifier(
#     base_score=0.5,
#     booster='gbtree',
#     colsample_bylevel=1,
#     learning_rate=0.1,
#     gamma=0,
#     max_delta_step=0,
#     max_depth=3,
#     min_child_weight=1,
#     missing=None,
#     n_estimators=100,
#     n_jobs=1,
#     nthread=None,
#     objective='binary:logistic',
#     random_state=0,
#     reg_alpha=0,
#     reg_lambda=1,
#     scale_pos_weight=1,
#     seed=None,
#     silent=None,
#     subsample=1,
#     verbosity=1
# )
# model.fit(x_train,y_train)

y_pred=grid_search.predict(x_test)
print(accuracy_score(y_test,y_pred)*100)










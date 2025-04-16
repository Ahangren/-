import numpy as np
import pandas as pd
from  datetime import datetime

from pandas.core.common import random_state
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import ElasticNetCV,LassoCV,RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import os

from 机器学习基础.Machine_learing_project.Detecting_fake_news import x_train

print(os.listdir("../input"))
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

print("Train set size: ",train.shape)
print("Test set size: ",test.shape)
print('start data precessing: ',datetime.now())

train.drop(['Id'],axis=1,inplace=True)
test.drop(['Id'],axis=1,inplace=True)

train=train[train.GrLivArea<4500]
train.reset_index(drop=True,inplace=True)

# 对预测目标数值进行对数变换和特征矩阵对象的创建
# log1p就是log（1+x），用来对房价数据进行数据预处理，他的好处是转化后的数据更加服从高斯分布
# 需要注意的是，最后需要将预测出的平滑数据还原，二还原过程就是log1p的逆运算expm1
train['SalePrice'] = np.log1p(train['SalePrice'])
# 单独取出训练数据中的房价信息，存入y对象
y=train['SalePrice'].reset_index(drop=True)
# 沿着水平方向找出列名为SalePrice的列，将它们全部删除，得到纯的feature数据
train_features=train.drop(['SalePrice'],axis=1)
# test中就不需要了，因为他本来就没有
test_features=test

# 合并训练数据特征矩阵和测试数据特征矩阵，以便统一进行特征处理
features=pd.concat([train_features,test_features]).reset_index(drop=True)
print(features.shape)
features['MSSubClass','YtSold','MoSold']=features['MSSubClass','YtSold','MoSold'].astype(str)

# 填充空值
features['Functional']=features['Functional'].fillna('Typ')
features['Electrical']=features['Electrical'].fillna('SBrkr')
features['KitchenQual']=features['KitchenQual'].fillna('TA')
features['PoolQC']=features['PoolQC'].fillna('None')

features['Exterior1st']=features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
features['Exterior2nd']=features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
features['SaleType']=features['SaleType'].fillna(features['SaleType'].mode()[0])

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    features[col]=features[col].fillna(0)

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    features[col]=features[col].fillna('None')

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    features[col] = features[col].fillna('None')


features['MSZoning']=features.groupby('MSSubClass')['MSZoning'].transform(lambda x:x.fillna(x.mode()[0]))

numeric_dtype=['int16','int32','int64','float16','float32','float64']
numerics=[]
for i in features.columns:
    if features[i].dtype in numeric_dtype:
        numerics.append(i)

features.update(features[numerics].fillna[0])


numeric_dtype2=['int16','int32','int64','float16','float32','float64']
numerics2=[]
for i in features.columns:
    if features[i].dtype in numeric_dtype2:
        numerics2.append(i)
skew_features=features[numerics2].apply(lambda x:skew(x)).sotr_values(ascending=False)

high_skew=skew_features[skew_features>0.5]
skew_index=high_skew.index

for i in skew_index:
    features[i]=boxcox1p(features[i],boxcox_normmax(features[i]+1))


features=features.drop(['Utilities','Street','PoolQC'],axis=1)

# 融合多个特征，生成新特征
features['YrBltAndRemod']=features['YearBuilt']+features['YearRemodAdd']
features['TotalSF']=features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']

features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +
                                 features['1stFlrSF'] + features['2ndFlrSF'])

features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +
                               features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))

features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +
                              features['EnclosedPorch'] + features['ScreenPorch'] +
                              features['WoodDeckSF'])


features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

print("删除了3个特征，又融合创建了10个新特征，处理之后的特征矩阵维度为:",features.shape)

# 特征投影
final_features=pd.get_dummies(features).reset_index(drop=True)
print("使用get_dummies()方法“投影”特征矩阵，即分解出更多特征，得到更多列。投影后的特征矩阵维度为:",final_features.shape)

X=final_features.iloc[:len(y),:] # y是列向量，存储了训练数据中的房价列信息，截取后得到的x阵的维度是len(y)*(final_features的列数)
X_sub=final_features.iloc[len(y):,:]
print("删除了3个特征，又融合创建了10个新特征，处理之后的特征矩阵维度为:",'X', X.shape, 'y', y.shape, 'X_sub', X_sub.shape)

outliers=[30,88,462,631,1322]
X=X.drop(X.index[outliers])
y=y.drop(y.index[outliers])

overfit=[]
for i in X.columns:
    counts=X[i].value_counts()
    zeros=counts.iloc[0]
    if zeros/len(X)*100>99.94:
        overfit.append(i)

overfit=list(overfit)
X=X.drop(overfit,axis=1)

X_sub=X_sub.drop(overfit)

print("删除极端值及过拟合列后，训练数据特征矩阵的维数为，特征：",'X', X.shape, '对应于特征的对数变换后的房价y', y.shape, '测试数据的特征矩阵（它应该在行、列上未被删减）X_sub', X_sub.shape)
print('特征处理已经完成。开始对训练数据进行机器学习', datetime.now())

Kfolds=KFold(n_splits=5, shuffle=True, random_state=42)

def rmsle(y,y_pred):
    return np.sqrt(mean_squared_error(y,y_pred))

def cv_rmse(model,X=X):
    rmse=np.sqrt(-cross_val_score(model,X,y,scoring='neg_mean_squared_error',cv=Kfolds))

alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

ridge=make_pipeline(RobustScaler(),RidgeCV(alphas=alphas_alt,cv=Kfolds))

lasso=make_pipeline(RobustScaler(),LassoCV(max_iter=1e7,alphas=alphas2,random_state=42,cv=Kfolds))

elasticnet=make_pipeline(RobustScaler(),ElasticNetCV(max_iter=1e7,alphas=e_alphas,cv=Kfolds,l1_ratio=e_l1ratio))

svr=make_pipeline(RobustScaler(),SVR(C=20,epsilon=0.008,gamma=0.003))

gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42)

lightgbm = LGBMRegressor(objective='regression',
                                       num_leaves=4,
                                       learning_rate=0.01,
                                       n_estimators=5000,
                                       max_bin=200,
                                       bagging_fraction=0.75,
                                       bagging_freq=5,
                                       bagging_seed=7,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=7,
                                       verbose=-1,
                                       #min_data_in_leaf=2,
                                       #min_sum_hessian_in_leaf=11
                                       )

xgboost = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)

stack_gen=StackingCVRegressor(regressors=(ridge,lasso,elasticnet,gbr,xgboost,lightgbm,svr),
                              meta_regressor=xgboost,use_features_in_secondary=True)


score = cv_rmse(ridge)
print("二范数rideg岭回归模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(lasso)
print("一范数LASSO收缩模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

#打印elastic net弹性网络模型的得分
score = cv_rmse(elasticnet)
print("elastic net弹性网络模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

#打印SVR支持向量机模型的得分
score = cv_rmse(svr)
print("SVR支持向量机模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

#打印lightgbm轻梯度提升模型的得分
score = cv_rmse(lightgbm)
print("lightgbm轻梯度提升模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

#打印gbr梯度提升回归模型的得分
score = cv_rmse(gbr)
print("gbr梯度提升回归模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

#打印xgboost模型的得分
score = cv_rmse(xgboost)
print("xgboost模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )


print(datetime.now(), '对stack_gen集成器模型进行参数训练')
stack_gen_model = stack_gen.fit(np.array(X), np.array(y))

print(datetime.now(), '对elasticnet弹性网络模型进行参数训练')
elastic_model_full_data = elasticnet.fit(X, y)

print(datetime.now(), '对一范数lasso收缩模型进行参数训练')
lasso_model_full_data = lasso.fit(X, y)

print(datetime.now(), '对二范数ridge岭回归模型进行参数训练')
ridge_model_full_data = ridge.fit(X, y)

print(datetime.now(), '对svr支持向量机模型进行参数训练')
svr_model_full_data = svr.fit(X, y)

print(datetime.now(), '对GradientBoosting梯度提升模型进行参数训练')
gbr_model_full_data = gbr.fit(X, y)

print(datetime.now(), '对xgboost二阶梯度提升模型进行参数训练')
xgb_model_full_data = xgboost.fit(X, y)

print(datetime.now(), '对lightgbm轻梯度提升模型进行参数训练')
lgb_model_full_data = lightgbm.fit(X, y)

def blend_models_predict(X):
    return ((0.1 * elastic_model_full_data.predict(X)) + \
            (0.05 * lasso_model_full_data.predict(X)) + \
            (0.1 * ridge_model_full_data.predict(X)) + \
            (0.1 * svr_model_full_data.predict(X)) + \
            (0.1 * gbr_model_full_data.predict(X)) + \
            (0.15 * xgb_model_full_data.predict(X)) + \
            (0.1 * lgb_model_full_data.predict(X)) + \
            (0.3 * stack_gen_model.predict(np.array(X))))
print('融合后的训练模型对原数据重构时的均方根对数误差RMSLE score on train data:')
print(rmsle(y, blend_models_predict(X)))


print('使用测试集特征进行房价预测 Predict submission', datetime.now(),)
submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
submission.iloc[:,1] = np.floor(np.expm1(blend_models_predict(X_sub)))
print('融合其他优秀模型的预测结果 Blend with Top Kernals submissions', datetime.now(),)
sub_1 = pd.read_csv('../input/top-10-0-10943-stacking-mice-and-brutal-force/House_Prices_submit.csv')
sub_2 = pd.read_csv('../input/hybrid-svm-benchmark-approach-0-11180-lb-top-2/hybrid_solution.csv')
sub_3 = pd.read_csv('../input/lasso-model-for-regression-problem/lasso_sol22_Median.csv')
submission.iloc[:,1] = np.floor((0.25 * np.floor(np.expm1(blend_models_predict(X_sub)))) +
                                (0.25 * sub_1.iloc[:,1]) +
                                (0.25 * sub_2.iloc[:,1]) +
                                (0.25 * sub_3.iloc[:,1]))
q1 = submission['SalePrice'].quantile(0.005)
q2 = submission['SalePrice'].quantile(0.995)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)

submission.to_csv("House_price_submission.csv", index=False)
print('融合结果.csv文件输出成功 Save submission', datetime.now())







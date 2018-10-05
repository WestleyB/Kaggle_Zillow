import numpy as np
import pandas as pd
import time
from datetime import datetime

from zillow_functions import create_newFeatures, data_preprocessing, memory_reduce, create_special_feature
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import gc
from sklearn import linear_model
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

from zillow_stacking import rmsle_cv, mae_cv


#--- Preprocessing ---

seed = 42
n_features = 100

print('\nLoading data ...')
train =  pd.read_csv('../data/train_2016_v2.csv')
prop = pd.read_csv('../data/properties_2016.csv')
sample = pd.read_csv('../data/sample_submission.csv')
df_train = pd.merge(train, prop, on='parcelid', how='left')[:100000]
print('Shape train: {}'.format(df_train.shape))
del train; gc.collect()


print('\nData preprocessing ...')
df_train = data_preprocessing(df_train)


print('\nCreating new features ...')
df_train = create_newFeatures(df_train)

# New special feature
dates_model_col = ['transaction_year', 'transaction_month', 'yearbuilt', 'house_age']
df_train['spe_feature'], datesFeature_mod = create_special_feature(df_train[dates_model_col], df_train['logerror'].values)

# Month feature
month_avgs = df_train.groupby('transaction_month').agg(['mean'])['logerror', 'mean'].values - df_train['logerror'].mean()
month_model = LinearRegression().fit(np.arange(4, 13, 1).reshape(-1, 1), 
                                     month_avgs[3:].reshape(-1, 1))                       
df_train['super_month'] = month_model.predict(df_train['transaction_month'].values.reshape(-1, 1))


print('\nReducing consumption memory ...')
df_train = memory_reduce(df_train)


print('\nCreating training set ...')
x_train = df_train.drop(['parcelid', 'logerror'], axis=1)
y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)


print('\nFeatures selection ...')
x_col = list(x_train.columns)
lr = LinearRegression()
rfe = RFE(lr)
rfe.fit(x_train, y_train)
x_val = [x[1] for x in [x for x in sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), x_col), reverse=True) if x[0] > 0]]

train_columns = x_val[:n_features]
x_train = x_train[train_columns]
y_mean = np.mean(y_train)


print('\nFeatures scaling ...')
sc = StandardScaler()
x_train = sc.fit_transform(x_train)



#--- Models ---

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.1, random_state=seed))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.1, l1_ratio=.9, random_state=seed))

# KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

GBoost = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, criterion='mae',
                                   max_depth=4, max_features='sqrt', verbose=2,
                                   min_samples_leaf=15, min_samples_split=10, 
                                   subsample=0.8, loss='huber', random_state =seed)

model_xgb = xgb.XGBRegressor(learning_rate=0.031, max_depth=8, 
                             min_child_weight=0, n_estimators=1000,
                             objective='reg:linear', eval_metric='mae', base_score=y_mean,
                             gamma=0.2, subsample=0.8, silent=1,
                             random_state=seed, nthread=-1)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=32, metric='mae',
                              learning_rate=0.0025, n_estimators=720, max_depth=100,
                              max_bin=55, bagging_fraction=0.95,
                              bagging_freq=8, feature_fraction=0.85,
                              feature_fraction_seed=seed, bagging_seed=seed,
                              min_data_in_leaf=6, min_sum_hessian_in_leaf=11)


score = mae_cv(x_train, y_train, lasso, 5)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = mae_cv(x_train, y_train, ENet, 5)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# score = mae_cv(x_train, y_train, KRR, 5)
# print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = mae_cv(x_train, y_train, GBoost, 5)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# score = mae_cv(x_train, y_train, model_xgb, 5)
# print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# score = mae_cv(x_train, y_train, model_lgb, 5)
# print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))


#--- First Step ---

averaged_models = AveragingModels(models=(ENet, GBoost, lasso))

score = mae_cv(x_train, y_train, averaged_models, 5)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))




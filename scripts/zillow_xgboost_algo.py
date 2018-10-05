import numpy as np
import pandas as pd
import time
from datetime import datetime

from sami_function import missing_ratio
from zillow_functions import create_newFeatures, data_preprocessing, memory_reduce
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import gc
from sklearn import linear_model

seed = 42


def create_special_feature(X, y=None, model=None):
    if (model == None) and (y != None):
        reg = linear_model.LinearRegression()
        reg.fit (X, y)
    else:
        reg = model
    return reg.predict(X), reg



print('\nLoading data ...')

train =  pd.read_csv('../data/train_2016_v2.csv')
prop = pd.read_csv('../data/properties_2016.csv')
sample = pd.read_csv('../data/sample_submission.csv')

df_train = pd.merge(train, prop, on='parcelid', how='left')
print('Shape train: {}'.format(df_train.shape))

del train; gc.collect()

print('\nData preprocessing ...')

df_train = data_preprocessing(df_train)


print('\nCreating new features ...')

df_train = create_newFeatures(df_train)
# New special feature
dates_model_col = ['transaction_year', 'transaction_month', 'yearbuilt', 'house_age']
df_train['spe_feature'], datesFeature_mod = create_special_feature(df_train[dates_model_col], df_train['logerror'].values)

print('\nReducing consumption memory ...')

df_train = memory_reduce(df_train)

# print('\nDropping columns ...')
# col_2_drop = list(missing_ratio(df_train, plot=False).index)
# df_train = df_train.drop(col_2_drop, axis=1)
# MAE 0.05255990000000001 for 50 rounds  [:39]

#print("\nExtract month's bias ...")
#month_avgs = df_train.groupby('transaction_month').agg(['mean'])['logerror', 'mean'].values - df_train['logerror'].mean()

#from sklearn.linear_model import LinearRegression
#month_model = LinearRegression().fit(np.arange(4, 13, 1).reshape(-1, 1), 
#                                     month_avgs[3:].reshape(-1, 1))


print('\nCreating training set ...')

x_train = df_train.drop(['parcelid', 'logerror'], axis=1)  # , 'propertyzoningdesc', 'propertycountylandusecode'
y_train = df_train['logerror'].values

print(x_train.shape, y_train.shape)

y_mean = np.mean(y_train)
train_columns = x_train.columns

sc = StandardScaler()

#month_values = df_train['transaction_month'].values
#x_train = np.hstack([x_train, month_model.predict(month_values.reshape(-1, 1))])

x_train = sc.fit_transform(x_train)

del df_train; gc.collect()

# split = 80000
# x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]
# # x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.12, random_state=seed)

print('\nBuilding DMatrix...')

# d_train = xgb.DMatrix(x_train, label=y_train)
# d_valid = xgb.DMatrix(x_valid, label=y_valid)

# del x_train, x_valid; gc.collect()

d_train = xgb.DMatrix(x_train, y_train)

del x_train; gc.collect()

print('\nTraining ...')

params = {
    'eta': 0.03,
    'max_depth': 10,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'base_score': y_mean,
    'silent': 1,
    'seed': seed
}

#--- cross-validation ---
cv_result = xgb.cv(
                    params, 
                    d_train, 
                    nfold=10,
                    num_boost_round=1000,
                    early_stopping_rounds=100,
                    verbose_eval=10, 
                    show_stdv=False
                  )

num_boost_rounds = cv_result['test-mae-mean'].argmin()
mean_mae = cv_result['test-mae-mean'].min()

print("\n\tMAE {} for {} rounds".format(mean_mae, num_boost_rounds))


#--- train model ---
clf = xgb.train(dict(params), d_train, num_boost_round=num_boost_rounds)

# watchlist = [(d_train, 'train'), (d_valid, 'valid')]
# clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)


# del d_train, d_valid
del d_train

print('\nBuilding test set ...')

sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(prop, on='parcelid', how='left')

del prop, sample; gc.collect()

p_test = []
batch_size = 100000
for batch in range(batch_size, df_test.shape[0]+batch_size, batch_size):
    
    print('\nWorking batch {}'.format(batch))
    
    df_test_batch = df_test[batch-batch_size:batch].copy()
    
    print('\nData preprocessing ...')
    
    df_test_batch['rawcensustractandblock'] = df_test_batch.rawcensustractandblock.fillna(df_test.rawcensustractandblock.mode()[0])
    df_test_batch = data_preprocessing(df_test_batch)
    df_test_batch = df_test_batch.fillna(1)
    
    print('\nCreating new features ...')
    
    df_test_batch = create_newFeatures(df_test_batch)
    df_test_batch['spe_feature'], nawFeature_mod = create_special_feature(df_test_batch[dates_model_col], model=datesFeature_mod)
    
    print('\nReducing consumption memory ...')
    
    df_test_batch = memory_reduce(df_test_batch)

    x_test_batch = df_test_batch[train_columns]
    #x_test_batch = np.hstack([x_test_batch, np.zeros((x_test_batch.shape[0], 1))])
    x_test_batch = sc.transform(x_test_batch)
    
    del df_test_batch; gc.collect()

    d_test = xgb.DMatrix(x_test_batch)

    del x_test_batch; gc.collect()

    print('\nPredicting on test ...')

    p_test_batch = clf.predict(d_test)

    del d_test; gc.collect()
    
    [p_test.append(p) for p in p_test_batch]

i = 0
sub = pd.read_csv('../data/sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = p_test[i::6]
    i = i + 1

print('\nWriting csv ...')
sub.to_csv('../submissions/xgb_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False, float_format='%.4f')

print('\nPrediction available !!!')


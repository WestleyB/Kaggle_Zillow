import numpy as np
import pandas as pd

import time
from datetime import datetime

from sklearn.model_selection import train_test_split
import xgboost as xgb
import gc

from zillow_functions import create_newFeatures, memory_reduce, data_preprocessing


seed = 42

%%time
print('\nLoading data ...')

train =  pd.read_csv('../data/train_2016_v2.csv')
prop = pd.read_csv('../data/properties_2016.csv')
sample = pd.read_csv('../data/sample_submission.csv')

print('\nCreating new features ...')

df_train = pd.merge(train, prop, on='parcelid', how='left')

df_train = create_newFeatures(df_train)

# print('\nData preprocessing ...')

# df_train = data_preprocessing(df_train)

print('\nReducing consumption memory ...')

df_train = memory_reduce(df_train)

print('\nCreating training set ...')

x_train = df_train.drop(['parcelid', 'logerror'], axis=1)  # , 'propertyzoningdesc', 'propertycountylandusecode'
y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)

y_mean = np.mean(y_train)
train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = x_train[c].map({True: 1, 'Y': 1})
    x_train[c] = x_train[c].fillna(0).astype(np.int8)


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
    'max_depth': 6,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'base_score': y_mean,
    'silent': 1,
    'seed': seed
}

# watchlist = [(d_train, 'train'), (d_valid, 'valid')]
# clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)


#--- cross-validation ---
cv_result = xgb.cv(
                    params,
                    d_train,
                    nfold=10,
                    num_boost_round=10000,
                    early_stopping_rounds=100,
                    verbose_eval=10,
                    show_stdv=False,
                    seed=seed
                  )

num_boost_rounds = len(cv_result)
print('Boost round parameter : {}'.format(num_boost_rounds))

#--- train model ---
clf = xgb.train(dict(params), d_train, num_boost_round=num_boost_rounds)


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

    print('\nCreating new features ...')

    df_test_batch['rawcensustractandblock'] = df_test_batch.rawcensustractandblock.fillna(df_test.rawcensustractandblock.mode()[0])
    df_test_batch = df_test_batch.fillna(0)

    df_test_batch = create_newFeatures(df_test_batch)

    for c in df_test_batch.dtypes[df_test_batch.dtypes == object].index.values:
        df_test_batch[c] = df_test_batch[c].map({True: 1, 'Y': 1})
        df_test_batch[c] = df_test_batch[c].fillna(0).astype(np.int8)

    print('\nReducing consumption memory ...')

    df_test_batch = memory_reduce(df_test_batch)

    x_test = df_test_batch[train_columns]

    del df_test_batch; gc.collect()

    d_test = xgb.DMatrix(x_test)

    del x_test; gc.collect()

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
sub.to_csv('../submissions/xgb_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

print('\nPrediction available !!!')

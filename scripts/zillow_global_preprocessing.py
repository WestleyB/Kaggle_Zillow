import numpy as np
import pandas as pd
import time
from datetime import datetime

from zillow_functions import create_newFeatures, data_preprocessing, memory_reduce, create_special_feature
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import gc
from sklearn import linear_model
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")


def prepro():
    seed = 42
    n_features = 100

    print('\nLoading data ...')
    train =  pd.read_csv('../data/train_2016_v2.csv')
    prop = pd.read_csv('../data/properties_2016.csv')
    sample = pd.read_csv('../data/sample_submission.csv')
    train17 =  pd.read_csv('../data/train_2017.csv')
    prop17 = pd.read_csv('../data/properties_2017.csv')

    train = pd.concat([train, train17])
    prop = pd.concat([prop, prop17])

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

    df_train.to_csv('../data/train_processed.csv', sep=';', index=False)


if __name__ == '__main__':
    print("Launch Preprocessing ...")
    prepro()
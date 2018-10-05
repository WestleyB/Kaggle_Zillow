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
from sklearn.model_selection import GridSearchCV, KFold
# from sklearn.pipeline import Pipeline
from sklearn.externals import joblib


def xgboostGridsearch():
    seed = 42
    n_features = 100

    print('\nLoad preprocessed data ...')
    df_train =  pd.read_csv('../data/train_processed.csv', sep=';')


    print('\nCreating training set ...')
    x_train = df_train.drop(['parcelid', 'logerror'], axis=1)
    y_train = df_train['logerror'].values
    y_mean = np.mean(y_train)
    print(x_train.shape, y_train.shape)


    print('\nLaunching GridSearch ...')
    param_grid = {
                    'learning_rate': 0.03276175506252493, # np.arange(0.03, 0.035+0.0, 0.001).tolist(),
                    'max_depth': [8], # np.arange(7, 10, 1).tolist(),
                    'min_child_weight': np.arange(0, 1, 1).tolist(),
                    'gamma': np.arange(0, 1, 0.2).tolist(),
                    'subsample': [0.8],
                    'objective': ['reg:linear'],
                    'n_estimators': [1000],
                    # 'eval_metric': ['mae'],
                    'reg_alpha': np.arange(0.0, 1.0+0.0, 0.2).tolist(), 
                    'reg_lambda': np.arange(0.0, 1.0+0.0, 0.2).tolist(),
                    'base_score': [y_mean],
                    # 'silent': [1],
                    'seed': [seed]
                    }




    grid_clf = GridSearchCV(xgb.XGBRegressor(), 
                            param_grid,
                            scoring='neg_mean_absolute_error',
                            cv=KFold(n_splits=5, random_state=seed, shuffle=True),
                            error_score='mae',
                            verbose=2,
                            n_jobs=-1,
                            refit=True, 
                            return_train_score=True)

    grid_clf.fit(x_train, y_train)

    print('\nBest score : {}'.format(grid_clf.best_score_))

    print('\nBest estim : ')
    print(grid_clf.best_estimator_)

    print('\nBest params : ')
    print(grid_clf.best_params_)

    print('\nSave best params : ')
    joblib.dump(grid_clf.best_estimator_, 'xgboost_model.pkl')


if __name__ == '__main__':
    xgboostGridsearch()
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import xgboost as xgb


seed = 42

print('\nLoad preprocessed data ...')
df_train =  pd.read_csv('../data/train_processed.csv', sep=';')


print('\nCreating training set ...')
x_train = df_train.drop(['parcelid', 'logerror'], axis=1)
y_train = df_train['logerror'].values
y_mean = np.mean(y_train)
print(x_train.shape, y_train.shape)


def hyperopt_train_test(params):
    clf = xgb.XGBRegressor(**params)
    return cross_val_score(clf, x_train, y_train).mean()

space4knn = {
	'learning_rate': hp.uniform('learning_rate', 0.02, 0.04),
	'max_depth': hp.choice('max_depth', range(6, 13)),
	'min_child_weight': hp.uniform('min_child_weight', 0, 2),
	'gamma': hp.uniform('gamma', 0.0, 5.0),
	'subsample': hp.uniform('subsample', 0.5, 1.0),
	'objective': hp.choice('objective', ['reg:linear']),
	'n_estimators': hp.choice('n_estimators', range(500, 3000, 100)),
	'reg_alpha': hp.uniform('reg_alpha', 0.0, 2.0),
	'reg_lambda': hp.uniform('reg_lambda', 0.0, 2.0),
	'base_score': hp.choice('base_score', [0.0122520153048]),
	'seed': hp.choice('seed', [42])
}

def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': acc, 'status': STATUS_OK}

print('\nLaunching Tuning Hyperparameters ...')
trials = Trials()
best = fmin(f, space4knn, algo=tpe.suggest, max_evals=100, trials=trials)
print('best:')
print(best)

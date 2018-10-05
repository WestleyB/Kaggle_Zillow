import numpy as np
import pandas as pd


# Train set
df_train = pd.merge(df_train, prop, on='parcelid', how='left')

# Test set
sample.rename(index=str, columns={'ParcelId': 'parcelid'}, inplace=True)
df_test = sample.merge(prop, on='parcelid', how='left')

def preprocess(dataframe):
    col_done = []

    for c in dataframe.columns:
        if dataframe[c].dtype == 'object':
            if dataframe[c].nunique() <= 3:
                print(str(c) + " : "
                  + str(dataframe[c].nunique()) + " - "
                  + str(dataframe[c].isnull().sum()) + " - "
                  + str(dataframe[c].dtype) + " - "
                  + str(dataframe[c].unique())
                  )
                dataframe[c] = dataframe[c].map({True: 1, 'Y': 1})
                dataframe[c] = dataframe[c].fillna(0).astype(np.int8)
                col_done.append(c)
            if c == 'transactiondate':
                dataframe[c] = pd.to_datetime(dataframe[c])
                col_done.append(c)
            if c == 'propertycountylandusecode':
                dataframe[c].fillna('023A', inplace =True)
                col_done.append(c)
            if c == 'propertyzoningdesc':
                dataframe[c].fillna('UNIQUE', inplace =True)
                col_done.append(c)

    if df_train[c].dtype == 'float64':
        if c == 'logerror':
            ulimit = np.percentile(df_train[c].values, 99)
            llimit = np.percentile(df_train[c].values, 1)
            df_train.loc[df_train[c]>ulimit, [c]] = ulimit
            df_train.loc[df_train[c]<llimit, [c]] = llimit
            col_done.append(c)

        if c in ['taxamount', 'finishedsquarefeet6', 'finishedsquarefeet12', 'finishedsquarefeet13', 'finishedsquarefeet15', 'finishedsquarefeet50']:
            df_train[c] = df_train[c].fillna(df_train[c].mean(axis=0))
            ulimit = np.percentile(df_train[c].values, 99.5)
            llimit = np.percentile(df_train[c].values, 0.5)
            df_train.loc[df_train[c]>ulimit, [c]] = ulimit
            df_train.loc[df_train[c]<llimit, [c]] = llimit
            col_done.append(c)

        if 'typeid' in c:
            df_train[c] = df_train[c].fillna(df_train[c].mode()[0]).astype(np.int8)
            col_done.append(c)

    return dataframe

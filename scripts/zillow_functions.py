import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model


def data_preprocessing(dataframe):

    dataframe['bathroomcnt'] = dataframe['bathroomcnt'].fillna(1)
    dataframe['bedroomcnt'] = dataframe['bedroomcnt'].fillna(1)
    dataframe['decktypeid'] = dataframe['decktypeid'].fillna(0)
    dataframe['garagecarcnt'] = dataframe['garagecarcnt'].fillna(0)
    dataframe['yardbuildingsqft17'] = dataframe['yardbuildingsqft17'].fillna(0)
    dataframe['pooltypeid10'] = dataframe.pooltypeid10.fillna(0).astype(np.int8)
    dataframe['pooltypeid7'] = dataframe.pooltypeid7.fillna(0).astype(np.int8)
    dataframe['pooltypei2'] = dataframe.pooltypeid2.fillna(0).astype(np.int8)
    dataframe['heatingorsystemtypeid'] = dataframe['heatingorsystemtypeid'].fillna(13)
    dataframe['airconditioningtypeid'] = dataframe['airconditioningtypeid'].fillna(5)
    dataframe['yearbuilt'] = dataframe['yearbuilt'].fillna(dataframe.yearbuilt.max()).astype(np.int16)
    dataframe['storytypeid'] = dataframe['storytypeid'].fillna(dataframe['storytypeid'].mode()[0])
    
    dataframe['taxdelinquencyyear'] = dataframe['taxdelinquencyyear'].fillna(15).astype(np.int8)
    dataframe['taxdelinquencyyear'] = np.where(dataframe.taxdelinquencyyear < 18, 2000 + dataframe.taxdelinquencyyear.astype(np.int16), 1900 + dataframe.taxdelinquencyyear.astype(np.int16)).astype(np.int16)

    dataframe['taxamount'] = dataframe['taxamount'].fillna(dataframe['taxamount'].mean())

    dataframe['rawcensustractandblock'] = dataframe.rawcensustractandblock.fillna(dataframe.rawcensustractandblock.mode()[0])

    dataframe['architecturalstyletypeid'] = dataframe['architecturalstyletypeid'].fillna(dataframe['architecturalstyletypeid'].mode()[0])
    dataframe['typeconstructiontypeid'] = dataframe['typeconstructiontypeid'].fillna(dataframe['typeconstructiontypeid'].mode()[0])
    dataframe['buildingclasstypeid'] = dataframe['buildingclasstypeid'].fillna(dataframe['buildingclasstypeid'].mode()[0])

    for c in dataframe.columns:
        if 'squarefeet' in c or 'sqft' in c or 'size' in c or 'pooltypeid' in c or 'cnt' in c or 'nbr' in c or 'number' in c:
            dataframe[c] = dataframe[c].fillna(0)
    
    #--- drop out ouliers ---
#    if 'logerror' in dataframe.columns:
#        len_init = len(dataframe)
#        dataframe = dataframe[dataframe['logerror'] > -0.4 ].copy()
#        dataframe = dataframe[dataframe['logerror'] < 0.4 ].copy()
#        len_final = len(dataframe)
#        print('Removing train outliers :\nbefore: {}\nAfter: {}'.format(len_init, len_final))
    
    #--- replace ouliers ---
    for c in dataframe.columns:
        if c == 'logerror':
            ulimit = np.percentile(dataframe[c].values, 99)
            llimit = np.percentile(dataframe[c].values, 1)
            dataframe.loc[dataframe[c] > ulimit, [c]] = ulimit
            dataframe.loc[dataframe[c] < llimit, [c]] = llimit
            print('\n\tOutliers treated ...')
    

    if 'transactiondate' in dataframe.columns:
        dataframe['transactiondate'] =  pd.to_datetime(dataframe['transactiondate'])

    for c in dataframe.dtypes[dataframe.dtypes == object].index.values:

        if len(dataframe[c].unique()) <= 2:
            dataframe[c] = dataframe[c].map({True: 1, 'Y': 1})
            dataframe[c] = dataframe[c].fillna(0).astype(np.int8)
        elif c != 'transactiondate':
            # del dataframe[c]
            dataframe[c] = dataframe[c].fillna(-1)
            lbl = LabelEncoder()
            lbl.fit(list(dataframe[c].values))
            dataframe[c] = lbl.transform(list(dataframe[c].values)).astype(np.int16)
        else:
            pass
            
    return dataframe


def create_newFeatures(dataframe):
    """
    Create new features for Zillow dataframe

    """

    if 'transactiondate' in dataframe.columns:
        dataframe['transaction_year'] = dataframe.transactiondate.dt.year.astype(np.int16)
        dataframe['transaction_month'] = dataframe.transactiondate.dt.month.astype(np.int8)
        # dataframe['transaction_day'] = dataframe.transactiondate.dt.weekday.astype(np.int8)
        dataframe['transaction_quarter'] = dataframe.transactiondate.dt.quarter.astype(np.int8)
        del dataframe['transactiondate']
    else:
        df_date = pd.DataFrame({'transaction_year': [2016, 2016, 2016, 2017, 2017, 2017],
                             'transaction_month': [10, 11, 12, 10, 11, 12]})
        df_date['tmp'] = 1
        dataframe['tmp'] = 1
        dataframe = pd.merge(dataframe, df_date, on='tmp')
        del dataframe['tmp']
        dataframe['transaction_quarter'] = dataframe.transaction_year.astype(str)+'-'+dataframe.transaction_month.astype(str).apply(lambda x: ('00'+x)[-2:]) +'-01'
        dataframe['transaction_quarter'] = dataframe.transaction_quarter.astype(object)
        dataframe['transaction_quarter'] = pd.to_datetime(dataframe['transaction_quarter'])
        dataframe['transaction_quarter'] = dataframe.transaction_quarter.dt.quarter

                             
    dataframe['rawcensustractandblock_states'] = dataframe.rawcensustractandblock.astype(str).apply(lambda x: x[:1]).astype(np.int8)
    dataframe['rawcensustractandblock_countries'] = dataframe.rawcensustractandblock.astype(str).apply(lambda x: x[1:4]).astype(np.int8)
    dataframe['rawcensustractandblock_tracts'] = dataframe.rawcensustractandblock.astype(str).apply(lambda x: x[4:11]).astype(np.float64)
    dataframe['rawcensustractandblock_blocks'] = dataframe.rawcensustractandblock.astype(str).apply(lambda x: 0 if x[11:] == '' else x[11:]).astype(np.int8)
    
    #--- how old is the house? ---
    dataframe['house_age'] = dataframe['transaction_year'].astype(np.int16) - dataframe['yearbuilt'].astype(np.int16)

    #--- how many rooms are there? ---
    dataframe['tot_rooms'] = dataframe['bathroomcnt'] + dataframe['bedroomcnt']

    #--- does the house have A/C? ---
    dataframe['AC'] = np.where(dataframe['airconditioningtypeid']>0, 1, 0)

    #--- Does the house have a deck? ---
    dataframe['deck'] = np.where(dataframe['decktypeid']>0, 1, 0)
    dataframe.drop('decktypeid', axis=1, inplace=True)

    #--- does the house have a heating system? ---
    dataframe['heating_system'] = np.where(dataframe['heatingorsystemtypeid']>0, 1, 0)

    #--- does the house have a garage? ---
    dataframe['garage'] = np.where(dataframe['garagecarcnt']>0, 1, 0)

    #--- does the house come with a patio? ---
    dataframe['patio'] = np.where(dataframe['yardbuildingsqft17']>0, 1, 0)

    #--- does the house have a pool?
    dataframe['pool'] = dataframe['pooltypeid10'] | dataframe['pooltypeid7'] | dataframe['pooltypeid2']
    dataframe['pool'] = dataframe['pool'].map({True: 1, False: 0})

    #--- does the house have all of these? -> spa/hot-tub/pool, A/C, heating system , garage, patio
    dataframe['exquisite'] = dataframe['pool'] + dataframe['patio'] + dataframe['garage'] + dataframe['heating_system'] + dataframe['AC']

    #--- Features based on location ---
    dataframe['x_loc'] = np.cos(dataframe['latitude']) * np.cos(dataframe['longitude'])
    dataframe['y_loc'] = np.cos(dataframe['latitude']) * np.sin(dataframe['longitude'])
    dataframe['z_loc'] = np.sin(dataframe['latitude'])

    return dataframe


def memory_reduce(dataframe):

    #--- Memory usage of entire dataframe ---
    mem = dataframe.memory_usage(index=True).sum()
    print("\tInitial size {:.2f} MB".format(mem/ 1024**2))

    #--- List of columns that cannot be reduced in terms of memory size ---
    count = 0
    for c in dataframe.columns:
        if dataframe[c].dtype == object:
            count+=1
    print('\tThere are {} columns that cannot be reduced'.format(count))

    count = 0
    for c in dataframe.columns:

        if dataframe[c].dtype in ['int8', 'int16', 'int32', 'int64']:
            
            if (np.iinfo(np.int8).min < dataframe[c].min()) and (dataframe[c].max() < np.iinfo(np.int8).max):
                count+=1
                dataframe[c] = dataframe[c].fillna(0).astype(np.int8)
            
            if (np.iinfo(np.int16).min < dataframe[c].min()) and (dataframe[c].max() < np.iinfo(np.int16).max) and ((np.iinfo(np.int8).min > dataframe[c].min()) or (dataframe[c].max() > np.iinfo(np.int8).max)):
                count+=1
                dataframe[c] = dataframe[c].fillna(0).astype(np.int16)
            
            if (np.iinfo(np.int32).min < dataframe[c].min()) and (dataframe[c].max() < np.iinfo(np.int32).max) and ((np.iinfo(np.int16).min > dataframe[c].min()) or (dataframe[c].max() > np.iinfo(np.int16).max)):
                count+=1
                dataframe[c] = dataframe[c].fillna(0).astype(np.int32)
            
            if (np.iinfo(np.int64).min < dataframe[c].min()) and (dataframe[c].max() < np.iinfo(np.int64).max) and ((np.iinfo(np.int32).min > dataframe[c].min()) or (dataframe[c].max() > np.iinfo(np.int32).max)):
                count+=1
                dataframe[c] = dataframe[c].fillna(0).astype(np.int64)

        if dataframe[c].dtype in ['float16', 'float32', 'float64']:
            
            if (np.finfo(np.float16).min < dataframe[c].min()) and (dataframe[c].max() < np.finfo(np.float16).max):
                count+=1
                dataframe[c] = dataframe[c].fillna(0).astype(np.float16)
            
            if (np.finfo(np.float32).min < dataframe[c].min()) and (dataframe[c].max() < np.finfo(np.float32).max) and ((np.finfo(np.float16).min > dataframe[c].min()) or (dataframe[c].max() > np.finfo(np.float16).max)):
                count+=1
                dataframe[c] = dataframe[c].fillna(0).astype(np.float32)
            
            if (np.finfo(np.float64).min < dataframe[c].min()) and (dataframe[c].max() < np.finfo(np.float64).max) and ((np.finfo(np.float32).min > dataframe[c].min()) or (dataframe[c].max() > np.finfo(np.float32).max)):
                count+=1
                dataframe[c] = dataframe[c].fillna(0).astype(np.float64)

    print('\tThere are {} columns reduced'.format(count))

    #--- Let us check the memory consumed again ---
    mem = dataframe.memory_usage(index=True).sum()
    print("\tFinal size {:.2f} MB".format(mem/ 1024**2))

    return dataframe


def create_special_feature(X, y=None, model=None):
    if model is None:
        reg = linear_model.LinearRegression()
        reg.fit (X, y)
    else:
        reg = model
    return reg.predict(X), reg

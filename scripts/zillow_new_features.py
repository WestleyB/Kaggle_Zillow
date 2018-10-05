import pandas as pd
import numpy as np

def create_newFeatures(dataframe):
    """
    Create new features for Zillow dataframe

    """

    dataframe['transactiondate'] =  pd.to_datetime(dataframe['transactiondate'])
    dataframe['transaction_year'] = dataframe.transactiondate.dt.year.astype(np.int16)
    dataframe['transaction_month'] = dataframe.transactiondate.dt.month.astype(np.int8)
    dataframe['transaction_day'] = dataframe.transactiondate.dt.weekday.astype(np.int8)

    dataframe['rawcensustractandblock_states'] = dataframe.rawcensustractandblock.astype(str).apply(lambda x: x[:1]).astype(np.int8)
    dataframe['rawcensustractandblock_countries'] = dataframe.rawcensustractandblock.astype(str).apply(lambda x: x[1:4]).astype(np.int8)
    dataframe['rawcensustractandblock_tracts'] = dataframe.rawcensustractandblock.astype(str).apply(lambda x: x[4:11]).astype(np.float64)
    dataframe['rawcensustractandblock_blocks'] = dataframe.rawcensustractandblock.astype(str).apply(lambda x: 0 if x[11:] == '' else x[11:]).astype(np.int8)

    #--- how old is the house? ---
    dataframe['house_age'] = 2017 - dataframe['yearbuilt'].fillna(2016).astype(np.int16)

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
    dataframe['pooltypeid10'] = dataframe.pooltypeid10.fillna(0).astype(np.int8)
    dataframe['pooltypeid7'] = dataframe.pooltypeid7.fillna(0).astype(np.int8)
    dataframe['pooltypei2'] = dataframe.pooltypeid2.fillna(0).astype(np.int8)
    dataframe['pool'] = dataframe['pooltypeid10'] | dataframe['pooltypeid7'] | dataframe['pooltypeid2']

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
    print("Initial size {:.2f} MB".format(mem/ 1024**2))

    #--- List of columns that cannot be reduced in terms of memory size ---
    count = 0
    for c in dataframe.columns:
        if dataframe[c].dtype == object:
            count+=1
    print('There are {} columns that cannot be reduced'.format(count))

    count = 0
    for c in dataframe.columns:
        if dataframe[c].dtype != object:
            if((c != 'logerror')|(c != 'yearbuilt')|(c != 'xloc')|(c != 'yloc')|(c != 'zloc')):
                if ((dataframe[c].max() < 255) & (dataframe[c].min() > -255)):
                    count+=1
                    dataframe[c] = dataframe[c].fillna(0).astype(np.int8)
                if ((dataframe[c].max() > 255) & (dataframe[c].min() > -255)
                   & (dataframe[c].max() < 65535) & (dataframe[c].min() > 0)):
                    count+=1
                    dataframe[c] = dataframe[c].fillna(0).astype(np.int16)
                if ((dataframe[c].max() > 65535) & (dataframe[c].min() > 0)
                   & (dataframe[c].max() < 4294967295) & (dataframe[c].min() > 0)):
                    count+=1
                    dataframe[c] = dataframe[c].fillna(0).astype(np.int8)
    print('There are {} columns reduced'.format(count))

    #--- Let us check the memory consumed again ---
    mem = dataframe.memory_usage(index=True).sum()
    print("Final size {:.2f} MB".format(mem/ 1024**2))

    return dataframe

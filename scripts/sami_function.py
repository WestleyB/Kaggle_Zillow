import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

def missing_ratio(dataframe, n=30, plot = True) :
    
    '''
    Compute the ratio of missing values by column and plot the latter

    Options : plot = True to display plot or False to disable plotting

    Returns the missing ratio dataframe

    '''
    try :

        all_data_na = (dataframe.isnull().sum() / len(dataframe)) * 100
        all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:n]
        missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

        if plot :
            f, ax = plt.subplots(figsize=(15, 12))
            plt.xticks(rotation='90')
            sns.barplot(x=all_data_na.index, y=all_data_na)
            plt.xlabel('Features', fontsize=15)
            plt.ylabel('Percent of missing values', fontsize=15)
            plt.title('Percent missing data by feature', fontsize=15)

        return(missing_data)

    except ValueError as e :

        print("The dataframe has no missing values, ", e)

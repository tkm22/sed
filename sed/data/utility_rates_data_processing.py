
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import glob


# In[4]:


def read_csv():
    """
    Reads in all the benchmarking data and combines them into a single data frame
    """

    data = pd.concat([pd.read_csv(f)
                      for f in glob.glob('*.csv')], ignore_index=True, sort=False)
    data.drop(['eiaid'], inplace=True, axis=1)

    return data


def rename_columns(df):
    """
    Renames the columns of the data frame

    Parameters
    ----------
    df : pandas data frame
        The building benchmarking data for a city

    Returns
    ----------
    df
        The modified data frame
    """

    df.rename({'zip': 'ZIPCode',
               'utility_name': 'UtilityCompany',
               'state': 'State',
               'service_type': 'ServiceType',
               'ownership': 'Ownership',
               'commercial_rate': 'CommercialRate',
               'industrial_rate': 'IndustrialRate',
               'residential_rate': 'ResidentialRate'
               }, axis=1, inplace=True)

    return df


def generate_data():
    """
    Generates cleaned energy consumption data into local directory
    """

    data = read_csv()
    data = rename_columns(data)
    data.to_csv(
        r'C:/Users/cjros/DIRECT/Capstone/Data/Cleaned_Data/utility_rates.csv',
        index=False)


# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import glob


# In[5]:


def read_csv():
    """
    Reads in all the benchmarking data and combines them into a single data frame
    """

    data = pd.concat([pd.read_csv(f)
                      for f in glob.glob('*.csv')], ignore_index=True)

    return data


def get_census_number(df):
    """
    Transforms the census block number into a different format

    Parameters
    ----------
    df : pandas data frame
        The building benchmarking data for a city

    Returns
    ----------
    df
        The modified data frame
    """

    for i in range(len(df)):
        if len(str(df['CensusBlock'][i])) > 11:
            number = df['CensusBlock'][i]
            df.at[i, 'CensusBlock'] = number[7:]

    return df


def clean_data(data):
    """
    Cleans the energy benchmarking data frame

    Parameters
    ----------
    df : pandas data frame
        The building benchmarking data for a city

    Returns
    ----------
    df
        The modified data frame
    """

    data = get_census_number(data)
    data = data.drop_duplicates()

    return data


def generate_data():
    """
    Generates cleaned energy consumption data into local directory
    """

    data = read_csv()
    data = clean_data(data)
    data.to_csv(
        r'C:/Users/cjros/DIRECT/Capstone/Data/Cleaned_Data/income.csv',
        index=False)

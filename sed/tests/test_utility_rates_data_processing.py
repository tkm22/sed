
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import unittest
import utility_rates_data_processing as urdp
import glob


# In[7]:


test_df = pd.read_csv('test_utility_rate_data.csv')


# In[9]:


class TestUtilityRates(unittest.TestCase):

    def test_rename_columns(self):
        df = urdp.rename_columns(test_df)
        df.drop(['eiaid'], inplace=True, axis=1)

        column_names = ['Year',
                        'ZIPCode',
                        'UtilityCompany',
                        'State',
                        'ServiceType',
                        'Ownership',
                        'CommercialRate',
                        'IndustrialRate',
                        'ResidentialRate'
                        ]

        self.assertListEqual(column_names, list(df.columns))

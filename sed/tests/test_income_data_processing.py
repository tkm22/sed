
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import unittest
import income_data_processing as idp
import glob


# In[2]:


test_df = pd.read_csv('test_income_data.csv')


# In[ ]:


class TestIncome(unittest.TestCase):

    def test_clean_data(self):
        self.assertIsInstance(idp.clean_data(test_df), pd.DataFrame)

    def test_get_census_number(self):
        df = pd.DataFrame(data=['random-censusblock'], columns=['CensusBlock'])
        idp.get_census_number(df)
        self.assertEquals(df['CensusBlock'][0], 'censusblock')

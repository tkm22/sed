import numpy as np
import pandas as pd
import unittest
import utility_rates_data_processing as urdp
import glob
from sed import utility_rates_data_processing as edp

test_df = pd.read_csv('test_utility_rate_data.csv')


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

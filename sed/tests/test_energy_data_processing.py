import numpy as np
import pandas as pd
import unittest
import energy_data_processing as edp
import glob
from sed import energy_data_processing as edp

test_df = pd.read_csv('test_benchmarking_data.csv')


class TestEnergyConsumption(unittest.TestCase):

    #    def test_read_csv(self):
    #        self.assertIsInstance(edp.read_csv(), pd.DataFrame)

    def test_add_BuildingID(self):
        self.assertIn('BuildingID', edp.add_BuildingID(test_df).columns)

    def test_reorganize_columns(self):
        df = edp.add_BuildingID(test_df)
        df = edp.reorganize_columns(df)

        column_names = ['BuildingID',
                        'Year',
                        'City',
                        'State',
                        'ZIPCode',
                        'BuildingType',
                        'PrimaryPropertyUse',
                        'YearBuilt',
                        'PropertyGFA(sf)',
                        'SiteEUI(kBtu/sf)',
                        'SourceEUI(kBtu/sf)',
                        'SiteEUIWN(kBtu/sf)',
                        'SourceEUIWN(kBtu/sf)'
                        ]

        self.assertListEqual(column_names, list(df.columns))

    def test_create_buildingtypes(self):
        building_types = ['Commercial',
                          'Residential',
                          'Industrial',
                          'Specialty',
                          'Other']

        df = test_df
        edp.create_buildingtypes(df)

        for i in range(len(building_types)):
            self.assertIn(building_types[i], np.array(df['BuildingType']))

    def test_reduce_data(self):
        df = test_df
        edp.reduce_data(df)

        self.assertNotIsInstance(df['SiteEUI(kBtu/sf)'], type(float('nan')))

    def test_convert_to_int(self):
        df = test_df
        df.dropna(inplace=True)
        df.reset_index(inplace=True, drop=True)
        edp.convert_to_int(df)

        self.assertIsInstance(df['Year'][0], np.int64)
        self.assertIsInstance(df['ZIPCode'][0], np.int64)
        self.assertIsInstance(df['YearBuilt'][0], np.int64)

    def test_convert_to_float(self):
        df = test_df
        df.dropna(inplace=True)
        df.reset_index(inplace=True, drop=True)
        edp.convert_to_float(df)

        self.assertIsInstance(df['SiteEUI(kBtu/sf)'][0], np.float64)
        self.assertIsInstance(df['SiteEUIWN(kBtu/sf)'][0], np.float64)
        self.assertIsInstance(df['SourceEUI(kBtu/sf)'][0], np.float64)
        self.assertIsInstance(df['SourceEUIWN(kBtu/sf)'][0], np.float64)

    def test_reduce_zipcodes(self):
        df = pd.DataFrame(data=['12345-6789'], columns=['ZIPCode'])
        edp.reduce_zipcodes(df)
        self.assertEquals(df['ZIPCode'][0], 12345)

    def test_clean_data(self):
        self.assertIsInstance(edp.clean_data(test_df), pd.DataFrame)

    def compile_data(self):
        self.assertIsInstance(edp.compile_data(), pd.DataFrame)

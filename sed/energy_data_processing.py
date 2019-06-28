
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[21]:


def add_BuildingID(df):
    """
    Creates a unique ID for each building in a city

    Parameters
    ----------
    df : pandas data frame
        The building benchmarking data for a city

    Returns
    ----------
    df
        The modified data frame
    """

    df = df.assign(id=(df['UID']).astype('category').cat.codes)
    df.rename({'id': 'BuildingID'}, axis='columns', inplace=True)

    return df


def reorganize_columns(df):
    """
    Rearranges the columns in the data frame

    Parameters
    ----------
    df : pandas data frame
        The building benchmarking data for a city

    Returns
    ----------
    df
        The modified data frame
    """

    fixed_column_order = ['BuildingID',
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

    revised_column_order = []
    for i in range(len(fixed_column_order)):
        if fixed_column_order[i] in df.columns:
            revised_column_order.append(fixed_column_order[i])
        else:
            continue

    df = df[revised_column_order]

    return df


def create_buildingtypes(df):
    """
    Categorizes the property use type into one of five building categories

    Parameters
    ----------
    df : pandas data frame
        The building benchmarking data for a city

    Returns
    ----------
    df
        The modified data frame
    """

    df['BuildingType'] = None

    residential = [
        'Multifamily Housing',
        'Residence Hall/Dormitory',
        'Residential Care Facility',
        'Senior Care Community']
    commercial = [
        'Bank Branch',
        'Enclosed Mall',
        'Financial Office',
        'Fitness Center/Health Club/Gym',
        'Food Service',
        'Hotel',
        'Movie Theater',
        'Office',
        'Parking',
        'Performing Arts',
        'Repair Services (Vehicle, Shoe, Locksmith, etc.)',
        'Restaurant',
        'Retail Store',
        'Self-Storage Facility',
        'Strip Mall',
        'Supermarket/Grocery Store',
        'Wholesale Club/Supercenter']
    industrial = [
        'Distribution Center',
        'Manufacturing/Industrial Plant',
        'Non-Refrigerated Warehouse',
        'Refrigerated Warehouse']
    specialty = [
        'Adult Education',
        'College/University',
        'Fire Station',
        'Hospital (General Medical & Surgical)',
        'K-12 School',
        'Laboratory',
        'Library',
        'Medical Office',
        'Museum',
        'Outpatient Rehabilitation/Physical Therapy',
        'Police Station',
        'Pre-school/Daycare',
        'Prison/Incarceration',
        'Social/Meeting Hall',
        'Urgent Care/Clinic/Other Outpatient',
        'Wastewater Treatment Plant',
        'Worship Facility']
    other = [
        'Mixed Use Property',
        'Other',
        'Other - Education',
        'Other - Entertainment/Public Assembly',
        'Other - Lodging/Residential',
        'Other - Mall',
        'Other - Public Services',
        'Other - Recreation',
        'Other - Services',
        'Other - Specialty Hospital',
    ]

    for i in range(len(df)):
        if df['PrimaryPropertyUse'][i] in residential:
            df.at[i, 'BuildingType'] = 'Residential'
        elif df['PrimaryPropertyUse'][i] in commercial:
            df.at[i, 'BuildingType'] = 'Commercial'
        elif df['PrimaryPropertyUse'][i] in industrial:
            df.at[i, 'BuildingType'] = 'Industrial'
        elif df['PrimaryPropertyUse'][i] in specialty:
            df.at[i, 'BuildingType'] = 'Specialty'
        elif df['PrimaryPropertyUse'][i] in other:
            df.at[i, 'BuildingType'] = 'Other'


def reduce_data(df):
    """
    Removes rows with no building site EUI

    Parameters
    ----------
    df : pandas data frame
        The building benchmarking data for a city
    """

    for i in range(len(df)):
        if pd.isna(df['SiteEUI(kBtu/sf)'][i]):
            df.drop([i], axis=0, inplace=True)
        else:
            continue

    df.reset_index(inplace=True, drop=True)


def convert_to_int(df):
    """
    Converts particular columns into the int data type

    Parameters
    ----------
    df : pandas data frame
        The building benchmarking data for a city
    """

    column_names = ['Year', 'ZIPCode', 'YearBuilt']

    for i in range(len(column_names)):
        column = column_names[i]
        df[column] = pd.to_numeric(
            df[column].astype(str).str.replace(
                '-', ''), errors='coerce')

    for i in range(len(column_names)):
        column = column_names[i]
        df[column] = df[column].astype('int64')


def convert_to_float(df):
    """
    Converts particular columns into the float data type

    Parameters
    ----------
    df : pandas data frame
        The building benchmarking data for a city
    """

    fixed_column_names = ['SiteEUI(kBtu/sf)',
                          'SiteEUIWN(kBtu/sf)',
                          'SourceEUI(kBtu/sf)',
                          'SourceEUIWN(kBtu/sf)']

    revised_column_names = []
    for i in range(len(fixed_column_names)):
        if fixed_column_names[i] in df.columns:
            revised_column_names.append(fixed_column_names[i])
        else:
            continue

    for i in range(len(revised_column_names)):
        column = revised_column_names[i]
        df[column] = pd.to_numeric(
            df[column].astype(str).str.replace(
                ',', ''), errors='coerce')

    for i in range(len(revised_column_names)):
        column = revised_column_names[i]
        df[column] = df[column].astype('float64')


def reduce_zipcodes(df):
    """
    Converts ZIP codes into standard 5-digit format

    Parameters
    ----------
    df : pandas data frame
        The building benchmarking data for a city
    """

    for i in range(len(df)):
        if len(str(df['ZIPCode'][i])) > 5:
            zipcode = str(df['ZIPCode'][i])
            df.at[i, 'ZIPCode'] = int(zipcode[0:5])
        else:
            continue


def clean_data(df):
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

    df = add_BuildingID(df)
    create_buildingtypes(df)
    df = reorganize_columns(df)
    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)
    convert_to_int(df)
    convert_to_float(df)
    reduce_data(df)
    reduce_zipcodes(df)
    return df


def read_csv():
    """
    Reads in all the benchmarking data and combines them into a list
    """

    data_atlanta = pd.read_csv(
        'Atlanta_Building_Energy_Benchmarking_precleaned.csv')
    data_boston = pd.read_csv(
        'Boston_Building_Energy_Benchmarking_precleaned.csv')
    data_chicago = pd.read_csv(
        'Chicago_Building_Energy_Benchmarking_precleaned.csv')
    data_minneapolis = pd.read_csv(
        'Minneapolis_Building_Energy_Benchmarking_precleaned.csv')
    data_newyork = pd.read_csv(
        'NewYork_Building_Energy_Benchmarking_precleaned.csv')
    data_philadelphia = pd.read_csv(
        'Philadelphia_Building_Energy_Benchmarking_precleaned.csv')
    data_portland = pd.read_csv(
        'Portland_Building_Energy_Benchmarking_precleaned.csv')
    data_sanfrancisco = pd.read_csv(
        'SanFrancisco_Building_Energy_Benchmarking_precleaned.csv')
    data_seattle = pd.read_csv(
        'Seattle_Building_Energy_Benchmarking_precleaned.csv')
    data_washingtondc = pd.read_csv(
        'WashingtonDC_Building_Energy_Benchmarking_precleaned.csv')

    data = [data_atlanta, data_boston,
            data_chicago, data_minneapolis,
            data_newyork, data_philadelphia,
            data_portland, data_sanfrancisco,
            data_seattle, data_washingtondc
            ]

    return data


def compile_data(data):
    """
    Cleans the energy benchmarking data frames and compiles into one data frame

    Parameters
    ----------
    data : list of pandas data frames
        The building benchmarking data for all cities

    Returns
    ----------
    data_compiled
        All of the cleaned compiled into one data frame
    """

    for i in range(len(data)):
        data[i] = clean_data(data[i])

    data_compiled = pd.concat(data, ignore_index=True, sort=False)
    data_compiled = data_compiled.drop_duplicates()

    return data_compiled


def generate_data():
    """
    Generates cleaned energy consumption data into local directory
    """

    data = read_csv()
    data = compile_data(data)
    data.to_csv(
        r'C:\Users\cjros\DIRECT\Capstone\Data\Cleaned_Data\energy_consumption.csv',
        index_label='UID')

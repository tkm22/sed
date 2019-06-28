import pandas as pd 
import statsmodels.formula.api as smf


def linear_fit(df):
"""This is a function to determine a linear regression model relating the impact 
of various factors on energy consumption.
df: data frame of energy consumption and income/utility parameters. 
"""

fit_object = smf.ols(formula='Energy ~ Year + ZIPCode + BuildingType + PrimaryPropertyUse + YearBuilt + PropertyGFA + UtilityCompany + Ownership + ServiceType + ResidentialRate + CommercialRate + IndustrialRate + HouseholdIncome + HouseIncomeMOE', data=df)
ft = fit_object.fit()
ft.summary()

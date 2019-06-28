{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd \n",
    "import statsmodels.formula.api as smf\n",
    "\n",
    "\n",
    "def linear_fit(df)\n",
    "\"\"\"This is a function to determine a linear regression model relating the impact \n",
    "of various factors on energy consumption.\n",
    "df: data frame of energy consumption and income/utility parameters. \n",
    "\"\"\"\n",
    "\n",
    "fit_object = smf.ols(formula='Energy ~ Year + ZIPCode + BuildingType + PrimaryPropertyUse + YearBuilt + PropertyGFA + UtilityCompany + Ownership + ServiceType + ResidentialRate + CommercialRate + IndustrialRate + HouseholdIncome + HouseIncomeMOE', data=df)\n",
    "ft = fit_object.fit()\n",
    "ft.summary()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

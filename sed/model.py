from __future__ import absolute_import, division, print_function

import json
import pandas as pd;
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import category_encoders as ce

import lightgbm
import xgboost

import optuna
import hyperopt
import joblib

from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler, OneHotEncoder,\
OrdinalEncoder, QuantileTransformer, FunctionTransformer, LabelEncoder
from sklearn.model_selection import train_test_split, BaseCrossValidator, cross_validate, cross_val_score, train_test_split, \
                                    ShuffleSplit, GridSearchCV, RandomizedSearchCV, learning_curve, validation_curve
from sklearn.linear_model import LinearRegression, LassoLarsCV, LassoCV, MultiTaskElasticNetCV, Lasso, \
RidgeCV, Ridge, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline, make_pipeline, make_union, FeatureUnion
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, mutual_info_regression, RFECV, SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.compose import make_column_transformer, TransformedTargetRegressor, make_column_selector as selector
from sklearn.decomposition import PCA
from sklearn import set_config


from lightgbm.sklearn import LGBMRegressor
from collections import defaultdict
from joblib import dump, load

__all__ = ["load_dataset", "preprocess_demographic_data", "preprocess_SED_data",
        "model", "hyper_params_search_model", "evaluation", "features_importance"]

def load_dataset(filepath):
    """
    Reads shapefile and selects desired route.

    Parameters
    ----------
    filepath: string
        Path of dataset (.csv file)

    Returns
    -------
    df : pd.DataFrame
        DataFrame of data
    """
    df = pd.read_csv(filepath)
    return df


def preprocess_demographic_data(pop, drop_cols, pivot_idx, pivot_col,
    aggfunc='sum'):
    """
    Standard preprocessing procedure if pop data follows the format of
    experimental datasets:
    Drop the unwanted columns, then create a spreadsheet-style pivot table
    as a DataFrame by dropping the MultiIndex objects on the columns.

    Parameters
    ----------
    pop : DataFrame
        demographic dataset.
    drop_cols : str or list of strs
        columns to drop/delete.
    pivot_idx/pivot_col : str, column, Grouper, array or list of the previous
        index/column to aggregate.
    aggfunc : function, list of functions, dict, default numpy.mean
        function to aggregate on

    Returns
    -------
    pop : DataFrame
        Proprocessed demographic Dataset in an excel style pivot table without
        MultiIndex.

    """
    pop.drop(labels=drop_cols, axis=1, inplace=True)
    pop = pop.pivot_table(index=pivot_idx, columns=pivot_col, aggfunc=aggfunc)
    pop.columns = pop.columns.droplevel()
    return pop


def preprocess_SED_data(df, drop_cols, drop_0=True, drop_missing=True):
    """
    Standard preprocessing procedure if SED data follows the format of
    experimental datasets:
    Drop the unwanted columns, samples' energy consumption with 0 or/and
    samples with missing values if set True.

    Parameters
    ----------
    df : DataFrame
        Economic dataset, with prediction columns:['SiteEUIWN(kBtu/sf)',
        'SourceEUIWN(kBtu/sf)']
    drop_cols : str or list of strs
        Columns to drop/delete.
    drop_0/drop_missing : bool , default True
        Rows to drop/delete.

    Returns
    -------
    df_p : DataFrame
        Proprocessed SED Dataset.
    """
    zero_mask = (df['SiteEUIWN(kBtu/sf)']==0) | (df['SourceEUIWN(kBtu/sf)']==0)
    df_p = df[~zero_mask]
    df_p.drop(drop_cols, axis=1, inplace=True)
    df_p.dropna(inplace=True)
    return df_p


def model(estimator, tree_based=False, ordinal_encode_cols=['DataYear', 'YearBuilt'],
          categorical_encode=OneHotEncoder(sparse=False, handle_unknown='ignore'),
          categorical_cols=['ZipCode'], norm_scaler=MinMaxScaler(),
          feature_selection=VarianceThreshold(0), feature_reduction='drop',
          n_jobs=4, verbose=1):
    """
    Instantialize a pipeline model for auto feature engineering and predition.

    Parameters
    ----------
    estimator : Sklearn API estimator
        Instance of sklearn API estimator with set parameters.
    tree_based : bool, default False
        estimator is tree-based algorithm or not
    ordinal_encode_cols : list of str or pd.columns
        Columns for ordinal encode.
    categorical_encode : Sklearn API Encoder, default OneHotEncoder()
        categorical encode methods for categorical columns and ZipCode columns,
        or 'drop', 'passthrough' string to drop or passthrough the numerical columns.
    norm_scaler : sklearn API scaler or specified str, default MinMaxScaler()
        normalizer/scaler for all numerical columns, or 'drop', 'passthrough'
        string to drop or passthrough the numerical columns.
    feature_selection : sklearn API feature_selection functions, default VarianceThreshold()
        Filter methods of feature selection， or use 'drop'
        string to ignore the feature selection process.
    feature_reduction : sklearn API decomposition functions, default 'drop'
        decomposition methods for feature reduction, use 'drop'
        string to ignore the feature reduction process.
    n_job : int, default=None
        Number of jobs to run in parallel. None means 1 unless in a
        joblib.parallel_backend context. -1 means using all processors.
    verbose : integer
        Controls the verbosity: the higher, the more messages.
    Returns
    -------
    model : estimators/Pipeline
        Instance of sklearn API Pipeline. Only need to use .fit method and input
        X_train and y_train to train the model.
    """


    if tree_based:
        model = make_pipeline(

        make_column_transformer(# convert category to number for tree method, 'passthrough if it is not a tree'
                                (ce.OrdinalEncoder(), selector(dtype_include=['object', 'category'])),
                                remainder='passthrough',   # minmaxscaler() or passthrough if it is a tree
                                n_jobs=n_jobs, verbose=verbose),

    #    make_union(feature_selection, feature_reduction, n_jobs=n_jobs, verbose=verbose), # feature selection/extraction, use passthrough for now
    #    feature selection/reduction by filtering out 0 variance, True for regression

        MultiOutputRegressor(estimator)

        #    TransformedTargetRegressor(regressor=, transformer=)
        )

    else:
        model = make_pipeline(

        make_column_transformer(
                                (ce.OrdinalEncoder(), ordinal_encode_cols), # year data with int type
                                # Specified categorical cols (zipcode) to encode or passthrough if it is a tree
                                (categorical_encode, categorical_cols),
                                # (objects and category types) or passthrough if it is a tree
                                (categorical_encode, selector(dtype_include=['object', 'category'])),
                                remainder=norm_scaler,   # minmaxscaler() or passthrough if it is a tree
                                n_jobs=n_jobs, verbose=verbose),

        make_union(feature_selection, feature_reduction, n_jobs=n_jobs, verbose=verbose), # feature selection/extraction, use passthrough for now
            # feature selection/reduction by filtering out 0 variance, True for regression

        MultiOutputRegressor(estimator)

        #     TransformedTargetRegressor(regressor=, transformer=)
        )

    return model


def hyper_params_search_model(model, param_grid, cv=None, scoring=None, n_jobs=4, verbose=3):
    """
    Implementation of GridSearchCV in sklearn API.

    Parameters
    ----------
    model : sklearn API estimator
        model to tune.
    param_grid : dict or list of dictionaries
        Dictionary with parameters names (str) as keys and lists of parameter
        settings to try as values, or a list of such dictionaries, in which
        case the grids spanned by each dictionary in the list are explored.
        This enables searching over any sequence of parameter settings.
    scoring : str, callable, list/tuple or dict, default=None
        A single str (see The scoring parameter: defining model evaluation
        rules) or a callable (see Defining your scoring strategy from metric
        functions) to evaluate the predictions on the test set.

        For evaluating multiple metrics, either give a list of (unique)
        strings or a dict with names as keys and callables as values.

        NOTE that when using custom scorers, each scorer should return a
        single value. Metric functions returning a list/array of values can
        be wrapped into multiple scorers that return one value each.

        If None, the estimator’s score method is used.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy. Possible inputs for cv are:

        None, to use the default 5-fold cross validation,
        integer, to specify the number of folds in a (Stratified)KFold,
        CV splitter,
        An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and y is
        either binary or multiclass, StratifiedKFold is used.
        In all other cases, KFold is used.
    n_jobs : int, default=None
        Number of jobs to run in parallel. None means 1 unless in a
        joblib.parallel_backend context. -1 means using all processors.
    verbose : integer
        Controls the verbosity: the higher, the more messages.
    Returns
    -------
    model : Initialized GridSearchCV
        see above, detail see GridSearchCV from sklearn API.

    """
    model = GridSearchCV(model, param_grid=param_grid, cv=cv, n_jobs=n_jobs, verbose=verbose)
    return model


def evaluation(model, X_train, y_train, X_test, y_test):
    """
    Evaluate the fitted model, return a dictionary with MSE and R^2 score results.

    Parameters
    ----------
    model : fitted sklearn API estimator
        Fitted model for evaluation
    X_train : DataFrame or array
        Training set of features
    y_train : DataFrame or array
        Training set of responses
    X_test : DataFrame or array
        Test set of features
    y_test : DataFrame or array
        Test set of responses
    Returns
    -------
    results : dict
        dict contains y_test prediction, r2 score of training set, test set
        and trainCV set if applicable, and mean square error of test set.
    """
    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_trainCV = None
    best_params = 'default params of estimator'
    cv_results = None
    if hasattr(model, 'best_score_'):
        r2_trainCV = model.best_score_
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    if hasattr(model, 'cv_results_'):
        cv_results = model.cv_results_
    if hasattr(model, 'best_params_'):
        best_params = model.best_params_
    results={'prediction': y_pred, 'r2_train: ': r2_train, 'r2_trainCV': r2_trainCV, 'r2_test': r2, 'mse_test': mse, 'best_params': best_params}
    return results

def features_importance(model, features_name, responses_name):
    """
    Plot the features importance and return corresponding features importance
    DataFrame.

    Parameters
    ----------
    model : fitted sklearn API estimator
        Fitted model for visualization
    features_name : list of str or DataFrame.columns
        features name for retreiving features importance
    responses_name : list of str or DataFrame.columns
        responses name for retreiving features importance
    Returns
    -------
    ft : DataFrame
        featuers importance DataFrame
    """
    siteEUI = model.named_steps['multioutputregressor'].estimators_[0]
    sourceEUI = model.named_steps['multioutputregressor'].estimators_[1]
    lightgbm.plot_importance(siteEUI)
    lightgbm.plot_importance(sourceEUI)

    features = pd.DataFrame([siteEUI.feature_importances_, sourceEUI.feature_importances_], columns=features_name)
    ft = features.T
    ft.columns = responses_name
    return ft

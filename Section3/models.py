import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
import scipy.io
import xarray as xr
import cftime
import warnings

from torch.autograd import Variable
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

######################################################################################################################################################

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def mlptraining(training_data, testing_data):
    X_train = training_data.drop('Generation', axis=1)
    y_train = training_data['Generation']
    X_test = testing_data.drop('Generation', axis=1)
    y_test = testing_data['Generation']

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (50, 50, 50), (100, 100), (100, 100, 100)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.01, 0.1, 0.4, 0.8],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [20000],
        'early_stopping': [True],
        'validation_fraction': [0.2]
    }
    
    mlp = MLPRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error', verbose=2)
    grid_search.fit(X_train_split, y_train_split)
    best_mlp = grid_search.best_estimator_
    best_mlp.fit(X_train_split, y_train_split)
    y_train_pred = best_mlp.predict(X_train_scaled)
    y_test_pred = best_mlp.predict(X_test_scaled)
    
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    
    variance_y_train = np.var(y_train)
    variance_y_test = np.var(y_test)
    
    nmse_train = mse_train / variance_y_train
    nmse_test = mse_test / variance_y_test
    
    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
    mape_test = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    best_parameter = grid_search.best_params_

    return y_train_pred, y_test_pred, mse_train, mse_test, rmse_train, rmse_test, r2_train, r2_test, mape_train, mape_test, nmse_train, nmse_test


# def mlptraining(training_data , testing_data):
#     X_train = training_data.drop('Generation', axis=1)
#     y_train = training_data['Generation']
#     X_test = testing_data.drop('Generation', axis=1)
#     y_test = testing_data['Generation']

#     scaler = MinMaxScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)
#     # param_grid = {
#     #     'hidden_layer_sizes': [(10,), (20,), (20,30),(30,), (20,50), (30 , 50), (50,), (100,), (50, 50), (100, 50), (100, 100)],
#     #     'activation': ['relu', 'tanh'],
#     #     'solver': ['adam', 'sgd'],
#     #     'alpha': [0.0001, 0.001, 0.01, 0.1 , 0.3 , 0.5 , 0.8],
#     #     'learning_rate': ['constant', 'adaptive'],
#     #     'max_iter': [20000],
#     #     'early_stopping': [True],
#     #     'validation_fraction': [0.2]
#     # }
#     param_grid = {
#     'hidden_layer_sizes': [(50,), (100,), (50, 50), (50,50,50), (100, 100) , (100,100,100)],
#     'activation': ['relu', 'tanh'],
#     'solver': ['adam', 'sgd'],
#     'alpha': [0.0001, 0.01, 0.1 , 0.4, 0.8],
#     'learning_rate': ['constant', 'adaptive'],
#     'max_iter': [20000],
#     'early_stopping': [True],
#     'validation_fraction': [0.2]
#     }
    
#     mlp = MLPRegressor(random_state=42)
#     grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error', verbose=2)
#     grid_search.fit(X_train_split, y_train_split)
#     best_mlp = grid_search.best_estimator_
#     best_mlp.fit(X_train_split, y_train_split)
#     y_train_pred = best_mlp.predict(X_train_scaled)
#     y_test_pred = best_mlp.predict(X_test_scaled)
    
#     mse_train = mean_squared_error(y_train, y_train_pred)
#     mse_test = mean_squared_error(y_test, y_test_pred)
#     rmse_train = np.sqrt(mse_train)
#     rmse_test = np.sqrt(mse_test)
#     r2_train = r2_score(y_train, y_train_pred)
#     r2_test = r2_score(y_test, y_test_pred)
#     mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
#     mape_test = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
#     best_parameter = grid_search.best_params_

#     return y_train_pred, y_test_pred, mse_train, mse_test, rmse_train, rmse_test, r2_train, r2_test, mape_train, mape_test


# def mlptraining(training_data , testing_data):
#     X_train = training_data.drop('Generation', axis=1)
#     y_train = training_data['Generation']
#     X_test = testing_data.drop('Generation', axis=1)
#     y_test = testing_data['Generation']
    
#     # Scaling the data
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     # Further splitting training data into training and validation sets for early stopping
#     X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)
    
#     mlp = MLPRegressor(hidden_layer_sizes=(100,100,100), solver='sgd', alpha=0.001, learning_rate='adaptive',
#                        max_iter=20000, early_stopping=True, validation_fraction=0.2, random_state=42)
    
#     # Train the model
#     mlp.fit(X_train_split, y_train_split)
    
#     # Predictions
#     y_train_pred = mlp.predict(X_train_scaled)
#     y_test_pred = mlp.predict(X_test_scaled)
    
#     # Calculating evaluation metrics
#     mse_train = mean_squared_error(y_train, y_train_pred)
#     mse_test = mean_squared_error(y_test, y_test_pred)
#     rmse_train = np.sqrt(mse_train)
#     rmse_test = np.sqrt(mse_test)
#     r2_train = r2_score(y_train, y_train_pred)
#     r2_test = r2_score(y_test, y_test_pred)
#     mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
#     mape_test = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
#     return y_train_pred ,y_test_pred, mse_train , mse_test , rmse_train , rmse_test , r2_train , r2_test , mape_train , mape_test

# def mlptraining(training_data, testing_data):
#     X_train = training_data.drop('Generation', axis=1)
#     y_train = training_data['Generation']
#     X_test = testing_data.drop('Generation', axis=1)
#     y_test = testing_data['Generation']
    
#     # Check for and handle NaNs and infinite values
#     def clean_data(df, df_name):
#         before_rows = df.shape[0]
#         df = df.replace([np.inf, -np.inf], np.nan)
#         df = df.dropna()
#         after_rows = df.shape[0]
#         if before_rows != after_rows:
#             print(f"Cleaned {before_rows - after_rows} rows from {df_name} due to NaNs or infinities.")
#         return df
    
#     # Check and clean training data
#     X_train = clean_data(X_train, "X_train")
#     y_train = y_train.loc[X_train.index]
    
#     # Check and clean testing data
#     X_test = clean_data(X_test, "X_test")
#     y_test = y_test.loc[X_test.index]
    
#     # Ensure target variable y does not contain NaNs
#     if y_train.isnull().any():
#         print("y_train contains NaNs. Cleaning y_train...")
#         before_y_train = y_train.shape[0]
#         y_train = y_train.dropna()
#         X_train = X_train.loc[y_train.index]
#         after_y_train = y_train.shape[0]
#         print(f"Cleaned {before_y_train - after_y_train} rows from y_train.")
    
#     if y_test.isnull().any():
#         print("y_test contains NaNs. Cleaning y_test...")
#         before_y_test = y_test.shape[0]
#         y_test = y_test.dropna()
#         X_test = X_test.loc[y_test.index]
#         after_y_test = y_test.shape[0]
#         print(f"Cleaned {before_y_test - after_y_test} rows from y_test.")
    
#     # Scaling the data
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     # Further splitting training data into training and validation sets for early stopping
#     X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)
    
#     mlp = MLPRegressor(hidden_layer_sizes=(100,100,100), solver='sgd', alpha=0.001, learning_rate='adaptive',
#                        max_iter=20000, early_stopping=True, validation_fraction=0.2, random_state=42)
    
#     # Train the model
#     mlp.fit(X_train_split, y_train_split)
    
#     # Predictions
#     y_train_pred = mlp.predict(X_train_scaled)
#     y_test_pred = mlp.predict(X_test_scaled)
    
#     # Check predictions for NaNs or infinities
#     if np.isnan(y_train_pred).any() or np.isinf(y_train_pred).any():
#         print("y_train_pred contains NaNs or infinities.")
    
#     if np.isnan(y_test_pred).any() or np.isinf(y_test_pred).any():
#         print("y_test_pred contains NaNs or infinities.")
    
    
#     # Calculating evaluation metrics
#     mse_train = mean_squared_error(y_train, y_train_pred)
#     mse_test = mean_squared_error(y_test, y_test_pred)
#     rmse_train = np.sqrt(mse_train)
#     rmse_test = np.sqrt(mse_test)
#     r2_train = r2_score(y_train, y_train_pred)
#     r2_test = r2_score(y_test, y_test_pred)
#     mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
#     mape_test = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    
#     return y_train_pred, y_test_pred, mse_train, mse_test, rmse_train, rmse_test, r2_train, r2_test, mape_train, mape_test

# Example usage:
# training_data = pd.read_csv('path_to_training_data.csv')
# testing_data = pd.read_csv('path_to_testing_data.csv')
# results = mlptraining(training_data, testing_data)

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
import random


from torch.autograd import Variable
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from geopy.geocoders import OpenCage
from models import *

######################################################################################################################################################
##Directory
Curr_dir = os.getcwd()
Data_dir = Curr_dir + '/Data/'
Climate_dir = Data_dir+'ClimateData/'
isHR = True
print('HR: ')

##Datetime Processing
def ParseDate(ConDate):
    ConDate = str(ConDate)
    Year = ConDate[0:4]
    Month = ConDate[4:6]
    Day = ConDate[6:8]
    return Year+':'+Month+':'+Day
def ParseTime(Time):
    Time = int(int(Time)/100)
    if Time < 10:
        return '0'+str(Time)+':00:00'
    else:
        return str(Time)+':00:00'

def find_nearest(latitudes, longitudes, target_lat, target_lon):
    nearest_lat = latitudes[np.abs(latitudes - target_lat).argmin()]
    nearest_lon = longitudes[np.abs(longitudes - target_lon).argmin()]
    return nearest_lat, nearest_lon

def map_month_to_season(month):
    month = int(month)
    if 3 <= month <= 5:
        return 1
    elif 6 <= month <= 8:
        return 2
    elif 9 <= month <= 11:
        return 3
    elif month == 12 or 1<= month <= 2:
        return 4
        
def encode_month(month):
    angle = 2 * np.pi * month / 12
    return np.sin(angle), np.cos(angle)
    
def categorize_time_of_day(hour):
    if 5 <= hour < 12:
        return 0
    elif 12 <= hour < 18:
        return 1
    else:
        return 2

start_time = cftime.DatetimeNoLeap(2000, 1, 1)
end_time = cftime.DatetimeNoLeap(2020, 1, 1, 00, 0, 0)
date_index = pd.date_range(start='2000-01-01', end='2020-01-01 00:00:00', freq='6H')
date_index_no_leap = date_index[~((date_index.month == 2) & (date_index.day == 29))]

resampling_periods = {
    'Monthly': '30D',
    'Weekly': '7D',
    'Daily': '1D',
}
######################################################################################################################################################
if isHR:
    dataset1 = xr.open_dataset(Climate_dir+'HR_1.nc')
    dataset2 = xr.open_dataset(Climate_dir+'HR_2.nc')
    
    lat_Value = dataset1.coords['lat'].values
    lon_Value = dataset1.coords['lon'].values
    
    mat_data = scipy.io.loadmat(Climate_dir + '/Bias.mat')
    bc_mat = mat_data['bc_HR']
    bc = pd.DataFrame(bc_mat)
    bc.columns = lat_Value
    bc = bc.set_index(lon_Value)
    results_dir = 'results_HR'
    images_dir = 'images_HR'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
else:
    dataset1 = xr.open_dataset(Climate_dir+'LR_1.nc')
    dataset2 = xr.open_dataset(Climate_dir+'LR_2.nc')
    
    lat_Value = dataset1.coords['lat'].values
    lon_Value = dataset1.coords['lon'].values
    
    mat_data = scipy.io.loadmat(Climate_dir + '/Bias.mat')
    bc_mat = mat_data['bc_LR']
    bc = pd.DataFrame(bc_mat)
    bc.columns = lat_Value
    bc = bc.set_index(lon_Value)
    results_dir = 'results_LR'
    images_dir = 'images_LR'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
######################################################################################################################################################
KeyCountydf = pd.read_excel(Data_dir + 'ERCOT_KeyCounty_Coordinated.xlsx', engine='openpyxl')

WindProf_csv1 = pd.read_csv(Data_dir+'ERCOT_2000_2009.csv')
WindProf_csv1['DATE'] = WindProf_csv1['DATE'].apply(ParseDate)
WindProf_csv1['TIME'] = WindProf_csv1['TIME'].apply(ParseTime)
WindProf_csv1['DATETIME'] = pd.to_datetime(WindProf_csv1['DATE'] + ' ' + WindProf_csv1['TIME'], format='%Y:%m:%d %H:%M:%S')
WindProf_csv1 = WindProf_csv1.drop(['DATE', 'TIME'], axis=1)
WindProf_csv1 = WindProf_csv1.set_index('DATETIME')

WindProf_csv2 = pd.read_csv(Data_dir+'ERCOT_2010_2019.csv')
WindProf_csv2['DATE'] = WindProf_csv2['DATE'].apply(ParseDate)
WindProf_csv2['TIME'] = WindProf_csv2['TIME'].apply(ParseTime)
WindProf_csv2['DATETIME'] = pd.to_datetime(WindProf_csv2['DATE'] + ' ' + WindProf_csv2['TIME'], format='%Y:%m:%d %H:%M:%S')
WindProf_csv2 = WindProf_csv2.drop(['DATE', 'TIME'], axis=1)
WindProf_csv2 = WindProf_csv2.set_index('DATETIME')

WindProf_df = pd.concat([WindProf_csv1, WindProf_csv2], axis=0)
windfarms = [col for col in WindProf_df.columns if col != 'Date']
selected_windfarms = windfarms[:22]

for farm_name in selected_windfarms:
    print(farm_name)
    if ('SITE_00008' in farm_name) or ('SITE_00017' in farm_name):
        continue
    df = WindProf_df[[farm_name]].copy()
    df.index = WindProf_df.index
    Name = farm_name.split(':')[0]
    UL_ID = int(Name.split('_')[1])
    Capacity = float(farm_name.split('=')[1])
    df.rename(columns={farm_name:'Generation'}, inplace=True)
    specific_rows = KeyCountydf[KeyCountydf['UL ID'] == UL_ID]
    Lat, Lon = KeyCountydf.loc[specific_rows.index[0], ['Latitude', 'Longitude']]
    Lon_ = Lon+360
    Lat, Lon_ = find_nearest(lat_Value, lon_Value, Lat, Lon_)
    Coefficient = bc.loc[(Lon_) , (Lat)]
    
    dataset1_cut = (dataset1[['WSPD100' , 'TD2m' , 'RH2m']].sel(lon = Lon_, lat = Lat, method='nearest'))*Coefficient
    mask_feb29 = (dataset1_cut['time'].dt.month == 2) & (dataset1_cut['time'].dt.day == 29)
    dataset1_cut = dataset1_cut.where(~mask_feb29, drop=True)
    dataset1_cut_= dataset1_cut.sel(time=slice(start_time, end_time))
    
    dataset2_cut = (dataset2[['WSPD100' , 'TD2m' , 'RH2m']].sel(lon = Lon_, lat = Lat, method='nearest'))*Coefficient
    mask_feb29 = (dataset2_cut['time'].dt.month == 2) & (dataset2_cut['time'].dt.day == 29)
    dataset2_cut = dataset2_cut.where(~mask_feb29, drop=True)
    dataset2_cut_= dataset2_cut.sel(time=slice(start_time, end_time))

    dataset_names = ['dataset1_cut_', 'dataset2_cut_']
    datasets = [globals()[name] for name in dataset_names]
    columns_to_average = ['WSPD100', 'TD2m', 'RH2m']
    average_dataset = pd.DataFrame(columns=columns_to_average)
    for column in columns_to_average:
        average_dataset[column] = sum(df[column] for df in datasets) / len(datasets)
    average_dataset.index = date_index_no_leap
    interpolated_Data = average_dataset.resample('H').interpolate(method='linear')

    df['WSPD100'] = interpolated_Data['WSPD100']
    df['TD2m'] = interpolated_Data['TD2m']
    df['RH2m'] = interpolated_Data['RH2m']
    df.loc[:, 'Month'] = df.index.month 
    df['Season'] = df['Month']
    df['Season'] = df['Season'].apply(map_month_to_season)
    df['Hour'] = df.index.hour
    df['Time_of_Day'] = df['Hour'].apply(categorize_time_of_day)

    for period_name, period in resampling_periods.items():
        if period_name != 'Hourly':
            df_windfarm = df.drop(['Hour' , 'Time_of_Day'] , axis = 1)
            df_windfarm = df_windfarm.resample(period).mean()
            correlations = df_windfarm.corr()
            generation_correlations = correlations["Generation"]
        else:
            correlations = df.corr()
            generation_correlations = correlations["Generation"]

        print(f"{period_name} Correlations for {farm_name}:\n", generation_correlations)
        
        training_data = df_windfarm['2000-01-01':'2014-12-31']
        testing_data = df_windfarm['2015-01-01':'2019-12-31']
        
        X_train = training_data.drop('Generation', axis=1)
        y_train = training_data['Generation']
        X_test = testing_data.drop('Generation', axis=1)
        y_test = testing_data['Generation']
        
        y_train_pred ,y_test_pred ,mse_train , mse_test , rmse_train , rmse_test , r2_train , r2_test , mape_train , mape_test , nmse_train, nmse_test = mlptraining(training_data , testing_data)
        result_text = (
            f"Windfarm: {farm_name}\n"
            f"Resampling Period: {period_name}\n"
            f"MSE for training data: {mse_train}\n"
            f"MSE for testing data: {mse_test}\n"
            f"NMSE for training data: {nmse_train}\n"
            f"NMSE for testing data: {nmse_test}\n"
            f"RMSE for training data: {rmse_train}\n"
            f"RMSE for testing data: {rmse_test}\n"
            f"R² score for training data: {r2_train}\n"
            f"R² score for testing data: {r2_test}\n"
            f"MAPE for training data: {mape_train}%\n"
            f"MAPE for testing data: {mape_test}%\n"
        )
        with open(f'{results_dir}/{farm_name}_{period_name}_results.txt', 'w') as f:
            f.write(result_text)

        plt.figure(figsize=(14, 6))

        # Plotting for training data
        plt.subplot(1, 2, 1)
        plt.plot(range(len(y_train)), y_train, label='True Values', color='blue', linestyle='-', marker='o', markersize=4)
        plt.plot(range(len(y_train)), y_train_pred, label='Predicted Values', color='red', linestyle='--', marker='x', markersize=4)
        plt.xlabel('Time')
        plt.ylabel('Generation')
        plt.title('Training Data: True vs Predicted')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Plotting for testing data
        plt.subplot(1, 2, 2)
        plt.plot(range(len(y_test)), y_test, label='True Values', color='blue', linestyle='-', marker='o', markersize=4)
        plt.plot(range(len(y_test)), y_test_pred, label='Predicted Values', color='red', linestyle='--', marker='x', markersize=4)
        plt.xlabel('Time')
        plt.ylabel('Generation')
        plt.title('Testing Data: True vs Predicted')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.suptitle(f'True vs Predicted Values for Training and Testing Data ({farm_name} - {period_name})', fontsize=16)
        plt.subplots_adjust(top=0.88)
        plt.savefig(f'{images_dir}/{farm_name}_{period_name}_true_vs_predicted.png')
        plt.close()

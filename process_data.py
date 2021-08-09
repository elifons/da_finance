import pandas as pd
import numpy as np
import os
import os.path as op


def create_directory(logdir):
    try:
        os.makedirs(logdir)
    except FileExistsError:
        pass


def prepare_target_df(df):
    ''' Clean dataframe to create targets. 
        Remove any returns that don't have enough history so they don't count towards the labeling.
    '''
    copy_of_df = df.copy()
    for cols in df.columns:
        for i in range(240, len(df)):
            if df[cols].iloc[i-240:i].isnull().values.any():
                copy_of_df.iloc[i][cols] = np.nan
    return copy_of_df

def calculate_target_df(df):
    ''' Stock returns that are above the daily median are labeled as one, and zero otherwise.
        Returns a dataframe with the classification labels.
    '''
    new_df = prepare_target_df(df)
    median = new_df.median(axis=1)
    target_df = new_df.subtract(median, axis=0)
    target_df[target_df>=0] = 1
    target_df[target_df<0] = 0
    return target_df

def normalize_df(df):
    mean_ = np.nanmean(df.values[:750])
    std_ = np.nanstd(df.values[:750])
    return (df-mean_)/std_


def slice_test_dataset(df_X, df_target, dest_dir, sp):
    cols = df_X.columns
    index_list, dates = [], []
    X_list = []
    Y_list = []
    lookback = 240
    for i in range(lookback, len(df_X)):
        dates.append(df_X.index[i])
        for j,col in enumerate(cols):
            X = df_X[col][i-lookback:i].values
            Y = df_target[col][i]
            if np.isnan(X).any() or np.isnan(df_X[col].iloc[i]):
                continue
            else: 
                index_list.append([i-240, j])
                X_list.append(X)
                Y_list.append(Y)
    columns = np.array(df_X.columns)
    dates_array = np.array(dates)
    index_array = np.array(index_list)
    inference_dir = op.join(dest_dir, 'sp'+str(sp))
    X_test = np.array(X_list).reshape(-1,240,1)
    Y_test = np.array(Y_list).reshape(-1,1)
    create_directory(inference_dir)
    np.save(op.join(inference_dir, 'columns.npy'), columns)
    np.save(op.join(inference_dir, 'dates.npy'), dates_array)
    np.save(op.join(inference_dir, 'index_array.npy'), index_array)
    np.save(op.join(dest_dir, 'study_period_X_'+str(sp)+'_test.npy'), X_test)
    np.save(op.join(dest_dir, 'study_period_Y_'+str(sp)+'_test.npy'), Y_test)



def slice_dataset(df_X, df_target, cut_=None, sp=None):
    cols = df_X.columns
    X_list = []
    Y_list = []
    for i in range(cut_):
        for col in cols:
            X = df_X[col][i:i+240].values
            Y = df_target[col][i+240]
            if np.isnan(X).any() or np.isnan(Y):
                continue
            else:
                X_list.append(X)
                Y_list.append(Y)
    X_train = np.array(X_list).reshape(-1,240,1)
    Y_train = np.array(Y_list).reshape(-1,1) 
    np.save(op.join(dest_dir, 'study_period_X_'+str(sp)+'_train.npy'), X_train)
    np.save(op.join(dest_dir, 'study_period_Y_'+str(sp)+'_train.npy'), Y_train)


def create_dataset(df_, sp, dest_dir):    
    # select only the companies that existed at the beginning of testing period
    cols = df_.iloc[750].dropna().index.values
    df_X = df_[cols]
    target_df = calculate_target_df(df_X)
    normalized_df = normalize_df(df_X)
    slice_dataset(normalized_df[:750], target_df[:750], cut_=750-240, sp=sp)
    slice_test_dataset(normalized_df[750-240:],target_df[750-240:], dest_dir, sp)
    # return train_x, train_y, test_x, test_y
    
def process_dataset(dest_dir):
    df = pd.read_csv('full_df_stocks.csv', index_col='smDate', parse_dates=True)
    j = 0
    count = 0
    while count+1000 < len(df):
        df_ = df.iloc[count:count+1000]
        create_dataset(df_, j, dest_dir)
        count += 250
        j += 1


dest_dir = 'data'
create_directory(dest_dir)    
process_dataset(dest_dir)



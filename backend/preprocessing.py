import pandas as pd
import pickle


def date_processing(df, date_column="date"):
    
    df['year'] = pd.DatetimeIndex(df[date_column]).year
    df['month'] = pd.DatetimeIndex(df[date_column]).month
    df['day'] = pd.DatetimeIndex(df[date_column]).day
    df.drop(columns=date_column,inplace=True)
    
    return df


def scale_data(df):
    with open('models/scaler.pkl', 'rb') as file:
        scalar = pickle.load(file)
    df = scalar.transform(df)
    return df
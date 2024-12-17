import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml


def load_data(filepath : str) -> pd.DataFrame :
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath} :{e}")


def split_data(data : pd.DataFrame, test_size: float) -> tuple[pd.DataFrame,pd.DataFrame]:
    try:
        return train_test_split(data, test_size= test_size, random_state=42)
    except Exception as e:
        raise Exception(f"Error splittin data : {e}")



def save_data(df : pd.DataFrame, filepath: str) -> None:
    try:
        df.to_csv(filepath,index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {filepath} :{e}")
    

def main():
    data_filepath = "../data/drought_data_v1.csv"
    raw_data_path = os.path.join("../data","raw")
    try:
        data = load_data(data_filepath)
        test_size = 0.20
        train_data,test_data = split_data(data, test_size)

        os.makedirs(raw_data_path)

        save_data(train_data,os.path.join(raw_data_path,"train.csv"))
        save_data(test_data, os.path.join(raw_data_path,"test.csv"))
    except Exception as e:
        raise Exception(f"An error occurred :{e}")
    
if __name__ == "__main__":
    main()
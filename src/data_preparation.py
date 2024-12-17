import pandas as pd
import numpy as np
import os


def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}: {e}")


def drop_null(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()



def date_processing(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    try:
        df['year'] = pd.DatetimeIndex(df[date_column]).year
        df['month'] = pd.DatetimeIndex(df[date_column]).month
        df['day'] = pd.DatetimeIndex(df[date_column]).day
        df.drop(columns=[date_column], inplace=True)
        return df
    except Exception as e:
        raise Exception(f"Error during date processing: {e}")

def remove_outliers(df: pd.DataFrame, columns: list, threshold: float = 3) -> pd.DataFrame:
    try:
        for column in columns:
            if column in df.columns:
                mean = df[column].mean()
                std_dev = df[column].std()
                df = df[(df[column] <= mean + threshold * std_dev) & 
                        (df[column] >= mean - threshold * std_dev)]
        return df
    except Exception as e:
        raise Exception(f"Error during outlier removal: {e}")


def save_data(df: pd.DataFrame, filepath: str) -> None:
    try:
        df.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {filepath}: {e}")


def main():
    try:
        raw_data_path = "../data/raw/"
        processed_data_path = "../data/processed"

        train_data = load_data(os.path.join(raw_data_path, "train.csv"))
        test_data = load_data(os.path.join(raw_data_path, "test.csv"))

        # Drop NaN values
        train_data = drop_null(train_data)
        test_data = drop_null(test_data)

        # Process date columns (assuming date column exists)
        date_column = 'date' 
        train_data = date_processing(train_data, date_column)
        test_data = date_processing(test_data, date_column)

        # Remove outliers
        # Remove outliers
        columns_to_filter = [
            'PRECTOT', 'PS', 'QV2M', 'T2M', 'T2MDEW', 'T2MWET', 'T2M_MAX', 'T2M_MIN', 
            'T2M_RANGE', 'TS', 'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS10M_RANGE', 
            'WS50M', 'WS50M_MAX', 'WS50M_MIN', 'WS50M_RANGE'
        ]
        train_data = remove_outliers(train_data, columns_to_filter)
        test_data = remove_outliers(test_data, columns_to_filter)

        # Create processed data directory if it doesn't exist
        os.makedirs(processed_data_path, exist_ok=True)

        # Save the processed data
        save_data(train_data, os.path.join(processed_data_path, "train_processed.csv"))
        save_data(test_data, os.path.join(processed_data_path, "test_processed.csv"))

    except Exception as e:
        raise Exception(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
import pandas as pd
import os
import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer
import re
# Load dataset and return cleaned and preprocessed features and target
# TODO: use more of the columns, preprocess them in a different way,
#       or even include new features (e.g. from other datasets)

# columns: 'Poster_Link', 'Series_Title', 'Released_Year', 'Certificate',
#          'Runtime', 'Genre', 'IMDB_Rating', 'Overview', 'Meta_score', 'Director',
#          'Star1', 'Star2', 'Star3', 'Star4', 'Votes'

# columns used in template: 'Released_Year', 'Certificate', 'Runtime', 'Genre' (kind of',
# 'IMDB_Rating', 'Meta_score', 'Votes'

# Dictionary to standardize column names across different datasets
STANDARD_COLUMNS = {
    "Series_Title": "Title",
    "Released_Year": "Year",
    "No_of_Votes": "Votes",
    "Meta_score": "Metascore",
    "IMDB_Rating": "Rate",
    "Gross": "Gross",
    "Runtime": "Duration"
}

def get_prepared_data(data_path="data"):
        # Standard column names you want to use

    # Load raw data
    # this function tries to combine all .csv files in the data folder
    # it matches them up using the "Series_Title" column
    # if you want to use additional datasets, make sure they have a "Series_Title" column
    # if not, you will need additional logic to join the datasets
    # do not rename the column by hand, add code before this point to rename it
    # remember: we will not manually modify your datasets, so your code must do any formatting automatically
    
    # Load and combine all our CSV files
    data = get_raw_data(data_path)
    
    # First, check what columns we actually have
    print("Available columns:", data.columns.tolist())

    
     # Extract Gross from info column
    data['Gross'] = data['Info'].str.extract(r'(?:Gross:|Box Office:)\s*\$?([\d,]+)').iloc[:, 0]
    

    # Drop columns we don't need
    columns_to_drop = ["Unnamed: 0", "Title", "Certificate", "Description", "Cast", "Poster_Link", "Director", "Star1", "Star2", "Star3", "Star4", "Overview"]
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])


    # Remove ' min' and convert to integer
    data['Duration'] = data['Duration'].str.replace(' min', '').astype(int)   

    # Ensure Genre is a string and replace None values
    data['Genre'] = data['Genre'].fillna("Unknown").astype(str)

    # Convert Genre column into lists
    data['Genre'] = data['Genre'].str.split(', ')

    # Use MultiLabelBinarizer to encode Genre
    mlb = MultiLabelBinarizer()
    genre_dummies = pd.DataFrame(mlb.fit_transform(data['Genre']), columns=mlb.classes_, index=data.index)

    # Add encoded genres back to the dataset
    data = pd.concat([data, genre_dummies], axis=1)
    data = data.drop('Genre', axis=1)  # Drop original Genre column

    # Extract 'Gross' and 'Votes' from 'Info'
    data['Gross'] = data['Info'].str.extract(r'(?:Gross:|Box Office:)\s*\$?([\d,]+)').iloc[:, 0]
    data['Votes'] = data['Info'].str.extract(r'Votes:\s*([\d,]+)', expand=False)
    data['Gross'] = data['Gross'].fillna(0)
    data['Gross'] = data['Gross'].apply(lambda x: int(str(x).replace(",", "")) if not pd.isna(x) else 0)
    data['Votes'] = data['Votes'].str.replace(',', '').astype(float).fillna(0)

    # Apply log transformation
    data['Gross'] = data['Gross'].apply(lambda x: np.log1p(x))
    data['Votes'] = data['Votes'].apply(lambda x: np.log1p(x))

    # Process 'Rate' and 'Metascore'
    data['Rate'] = pd.to_numeric(data['Rate'], errors='coerce').fillna(data['Rate'].mean())
    data['Metascore'] = pd.to_numeric(data['Metascore'], errors='coerce').fillna(data['Metascore'].mean())

    # Drop 'Info' if not needed anymore
    data = data.drop('Info', axis=1)

    # One-hot encode remaining non-numeric columns (if any)
    non_numeric = data.select_dtypes(include=['object', 'category']).columns
    print("Non-numeric columns to encode:", non_numeric.tolist())
    data = pd.get_dummies(data, columns=non_numeric)

    # Prepare features and target
    target = data['Gross']
    features = data.drop('Gross', axis=1)

    # Convert to float32
    features = features.astype('float32')
    target = target.astype('float32')

    # Convert to tensors
    features = torch.tensor(features.values, dtype=torch.float32)
    target = torch.tensor(target.values.reshape(-1, 1), dtype=torch.float32)

    return features, target

def get_all_titles(data_path="data"):
    data = get_raw_data(data_path)
    return data["Series_Title"]

def clean_and_rename_columns(df):
    """ Standardizes column names across different CSV formats. """
    df = df.rename(columns=STANDARD_COLUMNS)

    # Ensure all required columns exist (fill missing ones with NaN)
    for col in STANDARD_COLUMNS.values():
        if col not in df.columns:
            df[col] = None  # Fill missing columns with NaN

    return df

def get_raw_data(path="data"):
    """ Reads all CSV files in the directory, merges them, and standardizes column names. """
    files = [f for f in os.listdir(path) if f.endswith(".csv")]
    data = pd.DataFrame()

    for file in files:
        df = pd.read_csv(os.path.join(path, file))
        df = clean_and_rename_columns(df)

        # Merge datasets on "Title"
        if data.empty:
            data = df
        else:
            data = data.merge(df, on="Title", how="outer")  # Outer join keeps all data

    # ✅ Fix duplicate columns (e.g., "Genre_x" and "Genre_y")
    for col in STANDARD_COLUMNS.values():
        if f"{col}_x" in data.columns and f"{col}_y" in data.columns:
            data[col] = data[f"{col}_x"].combine_first(data[f"{col}_y"])  # Use non-null values
            data.drop([f"{col}_x", f"{col}_y"], axis=1, inplace=True)  # Drop duplicates

    # ✅ Ensure 'Genre' exists in the final dataset
    if 'Genre' not in data.columns:
        data['Genre'] = None  # Avoid KeyError if missing

    return data
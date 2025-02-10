import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer
import re

# Load dataset and return cleaned and preprocessed features and target
# Expected original columns: 
# 'Poster_Link', 'Series_Title', 'Released_Year', 'Certificate',
# 'Runtime', 'Genre', 'IMDB_Rating', 'Overview', 'Meta_score', 'Director',
# 'Star1', 'Star2', 'Star3', 'Star4', 'Votes'
#
# Note: In your new CSV, instead of an "Info" column, you have:
#   - "Gross" (already provided) 
#   - "No_of_Votes" (to be used as Votes)
def get_prepared_data(data_path="data"):

    # Load raw data (this function combines all .csv files in the folder)
    data = get_raw_data(data_path)
    
    # First, check what columns we actually have
    print("Available columns:", data.columns.tolist())
    
    # Rename columns so that our processing is consistent.
    # This mapping renames:
    #   - 'Series_Title' to 'Title' (for merging consistency)
    #   - 'No_of_Votes' to 'Votes'
    #   - 'Runtime' to 'Duration'
    #   - 'IMDB_Rating' to 'Rate'
    #   - 'Meta_score' to 'Metascore'
    rename_dict = {
        'Series_Title': 'Title',
        'No_of_Votes': 'Votes',
        'Runtime': 'Duration',
        'IMDB_Rating': 'Rate',
        'Meta_score': 'Metascore'
    }
    data = data.rename(columns=rename_dict)
    print("Columns after renaming:", data.columns.tolist())
    
    # Drop columns that are not used in the original model
    columns_to_drop = ["Unnamed: 0", "Title", "Certificate", "Description", "Cast", "Poster_Link", "Overview", "Director", "Star1", "Star2", "Star3", "Star4"]
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
    
    # Now, the remaining columns should include:
    # 'Released_Year', 'Duration', 'Genre', 'Rate', 'Metascore', 'Votes', 'Gross'
    print("Columns after dropping unnecessary ones:", data.columns.tolist())
    
    # Clean Duration: remove ' min' and convert to integer
    data['Duration'] = data['Duration'].str.replace(' min', '').astype(int)
    
    # Process Genre: split by comma and create dummy variables via MultiLabelBinarizer
    data['Genre'] = data['Genre'].str.split(', ')
    mlb = MultiLabelBinarizer()
    genre_dummies = pd.DataFrame(mlb.fit_transform(data['Genre']),
                                 columns=mlb.classes_,
                                 index=data.index)
    data = pd.concat([data, genre_dummies], axis=1)
    data = data.drop('Genre', axis=1)
    
    # Process Gross (target):
    # Clean by removing commas and convert to integer, then apply log transformation.
    data['Gross'] = data['Gross'].fillna(0)
    data['Gross'] = data['Gross'].apply(lambda x: int(str(x).replace(",", "")) if not pd.isna(x) else 0)
    data['Gross'] = data['Gross'].apply(lambda x: np.log1p(x))
    
    # Process Votes:
    # Here, we use the provided Votes column (previously No_of_Votes) and clean it.
    data['Votes'] = data['Votes'].astype(str).str.replace(',', '').astype(float).fillna(0)
    data['Votes'] = data['Votes'].apply(lambda x: np.log1p(x))
    
    # Process Rate and Metascore: ensure numeric and fill missing values with the mean
    data['Rate'] = pd.to_numeric(data['Rate'], errors='coerce').fillna(data['Rate'].mean())
    data['Metascore'] = pd.to_numeric(data['Metascore'], errors='coerce').fillna(data['Metascore'].mean())
    
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
    
    # Convert to PyTorch tensors
    features = torch.tensor(features.values, dtype=torch.float32)
    target = torch.tensor(target.values.reshape(-1, 1), dtype=torch.float32)
    
    return features, target

def get_all_titles(data_path="data"):
    data = get_raw_data(data_path)
    return data["Series_Title"]

def get_raw_data(path="data"):
    # Read all CSV files in the directory
    import os
    files = os.listdir(path)
    data = pd.DataFrame()
    
    for file in files:
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(path, file))
            df = df.dropna()  # Drop rows with missing values
            
            # Check if the dataset has a 'Title' or 'Id' column for merging
            if 'Title' in df.columns:
                merge_key = 'Title'
            elif 'Id' in df.columns:
                merge_key = 'Id'
            else:
                raise ValueError(f"The dataset {file} does not contain a 'Title' or 'Id' column.")
            
            if data.empty:
                data = df
            else:
                data = data.merge(df, on=merge_key, how='inner')
    
    return data

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer
import re
# Load dataset and return cleaned and preprocessed features and target
# TODO: use more of the columns, preprocess them in a different way,
#       or even include new features (e.g. from other datasets)

# columns: 'Poster_Link', 'Series_Title', 'Released_Year', 'Certificate',
#          'Runtime', 'Genre', 'IMDB_Rating', 'Overview', 'Meta_score', 'Director',
#          'Star1', 'Star2', 'Star3', 'Star4', 'No_of_Votes'

# columns used in template: 'Released_Year', 'Certificate', 'Runtime', 'Genre' (kind of',
# 'IMDB_Rating', 'Meta_score', 'No_of_Votes'
def get_prepared_data(data_path="data"):

    # Load raw data
    # this function tries to combine all .csv files in the data folder
    # it matches them up using the "Series_Title" column
    # if you want to use additional datasets, make sure they have a "Series_Title" column
    # if not, you will need additional logic to join the datasets
    # do not rename the column by hand, add code before this point to rename it
    # remember: we will not manually modify your datasets, so your code must do any formatting automatically
    data = get_raw_data(data_path)
    
    # First, check what columns we actually have
    print("Available columns:", data.columns.tolist())

     # Extract Gross from info column
    data['Gross'] = data['Info'].str.extract(r'(?:Gross:|Box Office:)\s*\$?([\d,]+)').iloc[:, 0]
    

   # Inside get_prepared_data function
    columns_to_drop = ["Unnamed: 0", "Title", "Certificate", "Description", "Cast"]
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])


    # Remove ' min' and convert to integer
    data['Duration'] = data['Duration'].str.replace(' min', '').astype(int)   

    # Convert Genre to categorical
    data["Genre"] = data["Genre"].astype('category')
        # Split genres into lists
    data['Genre'] = data['Genre'].str.split(', ')

    # Initialize MultiLabelBinarizer
    mlb = MultiLabelBinarizer()

    # Fit and transform the Genre column
    genre_dummies = pd.DataFrame(mlb.fit_transform(data['Genre']), columns=mlb.classes_, index=data.index)

    # Concatenate the new genre columns with the data
    data = pd.concat([data, genre_dummies], axis=1)

    # Drop the original 'Genre' column
    data = data.drop('Genre', axis=1)

    # Clean Gross column
    data['Gross'] = data['Gross'].fillna(0)
    data["Gross"] = data["Gross"].apply(lambda x: int(str(x).replace(",", "")) 
                                       if not pd.isna(x) else 0)
    # Check for missing values in 'Rate' and 'Metascore'
    data['Rate'] = pd.to_numeric(data['Rate'], errors='coerce')
    data['Metascore'] = pd.to_numeric(data['Metascore'], errors='coerce')

    # Fill missing values if needed
    data['Rate'] = data['Rate'].fillna(data['Rate'].mean())
    data['Metascore'] = data['Metascore'].fillna(data['Metascore'].mean())

    # Get non-numeric columns
    non_numeric = data.select_dtypes(include=['object', 'category']).columns
    print("Non-numeric columns to encode:", non_numeric.tolist())

    # Convert all categorical to numeric
    data = pd.get_dummies(data, columns=non_numeric)
    
    # Verify all numeric
    assert all(data.dtypes != 'object'), "Found non-numeric columns"
    
    # Split features/target
    features = data.drop(columns=["Gross"])
    target = data["Gross"]

    # Convert to float32
    features = features.astype('float32')
    target = target.astype('float32')

    # Convert to tensors
    features = torch.tensor(np.array(features), dtype=torch.float32)
    target = torch.tensor(np.array(target).reshape(-1, 1), dtype=torch.float32)

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
            
            # Check if the dataset has a 'Title' or 'Id' column
            if 'Title' in df.columns:
                merge_key = 'Title'  # Use 'Title' as the key for merging
            elif 'Id' in df.columns:
                merge_key = 'Id'  # Fallback to 'Id' if 'Title' is not present
            else:
                raise ValueError(f"The dataset {file} does not contain a 'Title' or 'Id' column.")
            
            # Merge datasets on the selected key
            if data.empty:
                data = df
            else:
                data = data.merge(df, on=merge_key, how='inner')
    
    return data
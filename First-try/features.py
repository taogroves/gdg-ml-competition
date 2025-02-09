import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def preprocess_data(df):
    df = df.copy()

    # Extract numerical values from existing columns
    df['runtime'] = df['Duration'].str.extract('(\d+)').astype(float)
    df['votes'] = df['Info'].str.extract('Votes: ([\d,]+)').str.replace(',','').astype(float)
    df['gross'] = df['Info'].str.extract('\$(\d+\.\d+)M').astype(float) * 1000000
    df['score'] = df['Rate'].astype(float)
    df['metascore'] = pd.to_numeric(df['Metascore'], errors='coerce')
    df['year'] = df['Title'].str.extract('\((\d{4})\)').astype(float)

    # Feature engineering using available columns
    df["vote_score_ratio"] = df["votes"] / (df["score"] + 1)
    df["vote_year_ratio"] = df["votes"] / (df["year"] - df["year"].min() + 1)
    df["score_runtime_ratio"] = df["score"] / (df["runtime"] + 1)
    df["votes_per_year"] = df["votes"] / (df["year"] - df["year"].min() + 1)
    df["is_recent"] = (df["year"] >= df["year"].quantile(0.75)).astype(int)
    df["is_high_votes"] = (df["votes"] >= df["votes"].quantile(0.75)).astype(int)
    df["is_high_score"] = (df["score"] >= df["score"].quantile(0.75)).astype(int)
    df['director'] = df['Cast'].str.extract('Director: ([^|]+)')
    df['star'] = df['Cast'].str.extract('Stars: ([^|]+)')


    # Update categorical features to match dataset
    categorical_features = [
        "Certificate",
        "Genre",
        "director",
        "star"
    ]

    for feature in categorical_features:
        df[feature] = df[feature].astype(str)
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])

    numerical_features = [
        "runtime",
        "score",
        "year",
        "votes",
        "metascore",
        "vote_score_ratio", 
        "vote_year_ratio",
        "score_runtime_ratio",
        "votes_per_year",
        "is_recent",
        "is_high_votes",
        "is_high_score"
    ]

    # Create feature matrix
    X = df[numerical_features + categorical_features].copy()
    
    # Set target variable
    y = np.log1p(df['gross']) if 'gross' in df.columns else None

    return X, y



def prepare_features(df):
    # Add error handling
    try:
        X, y = preprocess_data(df)
        if X is None or y is None:
            raise ValueError("Failed to create features or target variable")
        return X, y
    except Exception as e:
        print(f"Error in feature preparation: {str(e)}")
        raise
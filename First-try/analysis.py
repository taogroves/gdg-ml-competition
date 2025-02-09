import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import zscore
import os
from prep_data import get_prepared_data
import torch

def create_feature_scatterplots(features, target, feature_names, suffix=''):
    """Create scatterplots for each feature against the target variable"""
    features_np = features.numpy()
    target_np = target.numpy().ravel()
    
    for i, feature_name in enumerate(feature_names):
        plt.figure(figsize=(10, 6))
        plt.scatter(features_np[:, i], target_np, alpha=0.5)
        plt.xlabel(feature_name)
        plt.ylabel('Gross (log-transformed)')
        plt.title(f'{feature_name} vs Gross')
        safe_name = feature_name.replace(' ', '_').replace('/', '_').lower()
        plt.savefig(f'plots/{safe_name}_scatter{suffix}.png')
        plt.close()

def detect_outliers(data, threshold=3):
    """Detect outliers using z-score method"""
    try:
        data_np = data.numpy().ravel()
        z_scores = zscore(data_np)
        outliers = np.where((z_scores > threshold) | (z_scores < -threshold))[0]
        return outliers
    except Exception as e:
        print(f"Error detecting outliers: {e}")
        return []

def save_cleaned_data(features, target, outliers, output_dir='cleaned_data'):
    """Save data with outliers removed"""
    # Convert to numpy arrays
    features_np = features.numpy()
    target_np = target.numpy()
    
    # Create mask for non-outlier indices
    mask = np.ones(len(target_np), dtype=bool)
    mask[outliers] = False
    
    # Remove outliers
    clean_features = features_np[mask]
    clean_target = target_np[mask]
    
    # Save cleaned data
    os.makedirs(output_dir, exist_ok=True)
    np.save(f'{output_dir}/features.npy', clean_features)
    np.save(f'{output_dir}/target.npy', clean_target)
    
    return torch.tensor(clean_features), torch.tensor(clean_target)

def analyze_data(features, target, feature_names):
    """Main analysis function with before/after plots"""
    try:
        os.makedirs('plots', exist_ok=True)
        
        # Create initial scatterplots
        print("Creating initial scatterplots...")
        create_feature_scatterplots(features, target, feature_names, suffix='before')
        
        # Detect and remove outliers
        print("Detecting outliers...")
        outliers = detect_outliers(target)
        print(f"Found {len(outliers)} outliers")
        
        # Save cleaned data
        clean_features, clean_target = save_cleaned_data(features, target, outliers)
        print(f"Cleaned data shape: {clean_features.shape}")
        
        # Create plots with cleaned data
        print("Creating cleaned scatterplots...")
        create_feature_scatterplots(clean_features, clean_target, feature_names, suffix='after')
        
        return clean_features, clean_target, outliers
    except Exception as e:
        print(f"Error in analysis: {e}")
        return None, None, []

if __name__ == "__main__":
    try:
        # Load data
        features, target = get_prepared_data()
        
        # Define feature names directly (since features is a tensor)
        feature_names = ['rating', 'duration', 'rate', 'metascore', 'year', 'score', 
                        'votes', 'writer', 'star', 'country', 'company', 'log_budget',
                        'budget_vote_ratio', 'budget_runtime_ratio', 'budget_score_ratio',
                        'vote_score_ratio', 'budget_year_ratio', 'is_recent', 'is_high_budget',
                        'is_high_votes', 'is_high_score', 'genre_Mystery', 'genre_Sci-Fi', 
                        'genre_Music', 'genre_Musical', 'genre_Romance', 'genre_Comedy', 
                        'genre_Biography', 'genre_Film-Noir', 'genre_War', 'genre_Thriller', 
                        'genre_Fantasy', 'genre_Sport', 'genre_Drama', 'genre_History', 
                        'genre_Adventure', 'genre_Action', 'genre_Horror', 'genre_Animation', 
                        'genre_Crime', 'genre_Western', 'genre_Family', 'director_impact']
        
        # Verify feature names length matches tensor shape
        assert len(feature_names) == features.shape[1], \
            f"Feature names length ({len(feature_names)}) doesn't match tensor shape ({features.shape[1]})"
        
        # Run analysis and get cleaned data
        clean_features, clean_target, outliers = analyze_data(features, target, feature_names)
        
        print(f"""
Analysis complete:
- Original samples: {len(target)}
- Outliers removed: {len(outliers)}
- Clean samples: {len(clean_target)}
- Plots saved in 'plots' directory
- Cleaned data saved in 'cleaned_data' directory
        """)
        
    except Exception as e:
        print(f"Error in main: {e}")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from prep_data import get_prepared_data
from pathlib import Path
import joblib
from sklearn.base import clone

# Added new imports
from sklearn.utils import resample
from scipy.stats import norm

def transform_monetary_values(X, y):
    """Additional transformations for monetary values"""
    df = pd.DataFrame(X)
    
    # Enhanced interaction terms
    if 'budget' in df.columns and 'votes' in df.columns:
        df['budget_votes'] = df['budget'] * df['votes'] / 1e6  # Scaled for stability
    if 'budget' in df.columns and 'score' in df.columns:
        df['budget_score'] = df['budget'] * (df['score'] / 10)  # Normalized score
        
    # New: Production quality metric
    if 'budget' in df.columns and 'duration' in df.columns:
        df['production_quality'] = df['budget'] / (df['duration'] + 1e-6)
    
    return df.values, y

def weighted_mape(y_true, y_pred):
    """Enhanced MAPE with dynamic weighting and clipping"""
    y_true_usd = np.expm1(y_true)
    y_pred_usd = np.expm1(y_pred)
    
    # Filter out extreme low values
    valid_mask = (y_true_usd > 1000) & (y_pred_usd > 1000)
    y_true_usd = y_true_usd[valid_mask]
    y_pred_usd = y_pred_usd[valid_mask]
    
    if len(y_true_usd) == 0:
        return np.nan  # Return NaN if no valid samples
    
    # Dynamic weighting using sigmoid function
    weights = 1 / (1 + np.exp(-y_true_usd/1e6))  # Scales with gross value
    weights /= weights.mean()  # Normalize weights
    
    # Calculate safe MAPE
    with np.errstate(divide='ignore', invalid='ignore'):
        errors = np.abs((y_true_usd - y_pred_usd) / y_true_usd)
    
    # Handle invalid values
    errors = np.nan_to_num(errors, nan=0.0, posinf=0.0, neginf=0.0)
    return np.average(errors, weights=weights)

def calculate_metrics(y_true, y_pred):
    # Convert to USD with clipping
    y_true_usd = np.expm1(np.clip(y_true, -1, None))
    y_pred_usd = np.expm1(np.clip(y_pred, -1, None))
    
    # High-value mask (top 5% instead of 10% for stricter selection)
    high_value_threshold = np.percentile(y_true_usd, 95)
    high_value_mask = y_true_usd > high_value_threshold
    
    # Only calculate if we have enough samples
    if np.sum(high_value_mask) > 10:
        high_value_mape = np.mean(np.abs(
            (y_true_usd[high_value_mask] - y_pred_usd[high_value_mask]) / 
            y_true_usd[high_value_mask])) * 100
    else:
        high_value_mape = np.nan
    
    return {
        'r2': r2,
        'mse': mse,
        'mape': valid_mape if np.any(mask) else 0,
        'weighted_mape': wmape,
        'msle': msle,
        'high_value_mape': high_value_mape
    }

def ensemble_predict(X, models):
    """Enhanced ensemble prediction with uncertainty estimation"""
    predictions = np.column_stack([model.predict(X) for model in models])
    return {
        'median': np.median(predictions, axis=1),
        'mean': np.mean(predictions, axis=1),
        'std': np.std(predictions, axis=1)
    }

def predict_with_confidence(X, model, confidence=0.95):
    """Improved confidence interval estimation"""
    # Create bootstrap samples
    bootstrap_preds = [model.predict(X) for _ in range(100)]
    predictions = np.median(bootstrap_preds, axis=0)
    pred_std = np.std(bootstrap_preds, axis=0)
    
    # Calculate confidence intervals using normal distribution
    z = norm.ppf((1 + confidence) / 2)
    lower = predictions - z * pred_std
    upper = predictions + z * pred_std
    
    return predictions, lower, upper

def analyze_feature_importance(model, feature_names):
    """Enhanced feature importance analysis"""
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[-20:]  # Show top 20 features
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    
    plt.figure(figsize=(12, 8))
    plt.barh(pos, importance[sorted_idx], align='center')
    plt.yticks(pos, np.array(feature_names)[sorted_idx])
    plt.xlabel("Relative Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig('results/feature_importance.png')
    plt.close()
    
    # Return full importance data
    return pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

def stratified_evaluation(X, y, model, budget_column_index):
    """Enhanced stratified evaluation with more bins and metrics"""
    try:
        budget_values = X[:, budget_column_index]
        percentiles = np.percentile(budget_values, [20, 40, 60, 80])
        
        results = []
        ranges = [
            'Very Low Budget', 'Low Budget', 
            'Medium Budget', 'High Budget', 
            'Very High Budget'
        ]
        
        for i, (low, high) in enumerate([
            (-np.inf, percentiles[0]),
            (percentiles[0], percentiles[1]),
            (percentiles[1], percentiles[2]),
            (percentiles[2], percentiles[3]),
            (percentiles[3], np.inf)
        ]):
            mask = (budget_values >= low) & (budget_values <= high)
            n_samples = np.sum(mask)
            
            if n_samples > 10:  # Minimum samples threshold
                X_tier = X[mask]
                y_tier = y[mask]
                preds = model.predict(X_tier)
                
                metrics = calculate_metrics(y_tier, preds)
                results.append({
                    'range': ranges[i],
                    'n_samples': n_samples,
                    **metrics
                })
                
                print(f"\n{ranges[i]} Results (n={n_samples}):")
                print(f"R²: {metrics['r2']:.4f}")
                print(f"MAPE: {metrics['mape']:.2f}%")
                print(f"High-value MAPE: {metrics['high_value_mape']:.2f}%")
            else:
                print(f"\n{ranges[i]}: Insufficient samples ({n_samples})")
        
        return pd.DataFrame(results)
    except Exception as e:
        print(f"Stratified evaluation error: {str(e)}")
        return None

def add_interactions(X, feature_names):
    """Enhanced interaction features with quadratic terms"""
    try:
        X_new = X.copy()
        important_features = [
            'vote_score_ratio', 'log_budget', 
            'budget_runtime_ratio', 'production_quality'
        ]
        interaction_names = []
        
        # Quadratic terms
        for f in important_features:
            if f in feature_names:
                idx = feature_names.index(f)
                X_new = np.column_stack([X_new, X[:, idx]**2])
                interaction_names.append(f"{f}_squared")
        
        # Cross interactions
        for i, f1 in enumerate(important_features):
            if f1 in feature_names:
                idx1 = feature_names.index(f1)
                for f2 in important_features[i+1:]:
                    if f2 in feature_names:
                        idx2 = feature_names.index(f2)
                        X_new = np.column_stack([
                            X_new, 
                            X[:, idx1] * X[:, idx2]
                        ])
                        interaction_names.append(f"{f1}_{f2}_interaction")
        
        return X_new, feature_names + interaction_names
    except Exception as e:
        print(f"Interaction error: {str(e)}")
        return X, feature_names

def train_specialized_models(X, y, budget_column_index, base_params):
    """Enhanced specialized models with early stopping"""
    try:
        budget_values = X[:, budget_column_index]
        percentiles = np.percentile(budget_values, [33, 66])
        
        models = {}
        ranges = {
            'low_budget': (-np.inf, percentiles[0]),
            'medium_budget': (percentiles[0], percentiles[1]),
            'high_budget': (percentiles[1], np.inf)
        }
        
        for range_name, (low, high) in ranges.items():
            mask = (budget_values >= low) & (budget_values <= high)
            X_sub = X[mask]
            y_sub = y[mask]
            
            if len(y_sub) > 100:  # Minimum sample size
                # Clone base model and add early stopping
                model = clone(base_params['estimator'])
                model.set_params(**{
                    'n_iter_no_change': 10,
                    'validation_fraction': 0.1,
                    **base_params['params']
                })
                
                model.fit(X_sub, y_sub)
                models[range_name] = model
                print(f"Trained {range_name} model with {len(y_sub)} samples")
            else:
                print(f"Insufficient samples for {range_name} model: {len(y_sub)}")
        
        return models
    except Exception as e:
        print(f"Specialized model error: {str(e)}")
        return None

def weighted_prediction(X, models, budget_column_index):
    """Enhanced prediction with smooth budget transitions"""
    try:
        budget_values = X[:, budget_column_index]
        percentiles = np.percentile(budget_values, [33, 66])
        
        predictions = np.zeros(len(X))
        
        for i, budget in enumerate(budget_values):
            if budget <= percentiles[0]:
                # Low budget model
                pred = models['low_budget'].predict(X[i:i+1])[0]
            elif budget <= percentiles[1]:
                # Medium budget blend
                low_pred = models['low_budget'].predict(X[i:i+1])[0]
                med_pred = models['medium_budget'].predict(X[i:i+1])[0]
                t = (budget - percentiles[0]) / (percentiles[1] - percentiles[0])
                pred = (1 - t)*low_pred + t*med_pred
            else:
                # High budget model
                pred = models['high_budget'].predict(X[i:i+1])[0]
            
            # Fix: Convert array to scalar using .item()
            predictions[i] = pred.item()  # CRITICAL FIX HERE
            
        return predictions
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None

# Main execution with improvements
if __name__ == "__main__":
    try:
                # In main execution section
        Path("saved_models").mkdir(exist_ok=True)  # Add this line
        Path("saved_weights").mkdir(exist_ok=True)
                
        print("Loading data...")
        features, target = get_prepared_data()
        X = features.numpy()
        y = target.numpy().ravel()
        
        # Apply log transformation to target
        y = np.log1p(y)  # CRITICAL IMPROVEMENT
        
        # Enhanced feature names with new interactions
        feature_names = [
            'rating', 'duration', 'rate', 'metascore', 'year', 'score', 
            'votes', 'writer', 'star', 'country', 'company', 'log_budget', 
            'budget_vote_ratio', 'budget_runtime_ratio', 'budget_score_ratio', 
            'vote_score_ratio', 'budget_year_ratio', 'is_recent', 'is_high_budget', 
            'is_high_votes', 'is_high_score', 'production_quality'
        ]
        genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 
                 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'History', 'Horror', 
                 'Music', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Sport', 
                 'Thriller', 'War', 'Western']
        feature_names.extend([f'genre_{genre}' for genre in genres])
        feature_names.append('director_impact')
        
        print("\nAdding enhanced interaction features...")
        X, feature_names = add_interactions(X, feature_names)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Optimized parameter grid
                # Update parameter grid
        param_grid = {
            "n_estimators": [300, 500],
            "max_depth": [3, 4],  # Prefer shallower trees
            "learning_rate": [0.01, 0.05],
            "min_samples_split": [20, 30],
            "min_samples_leaf": [10, 15],
            "subsample": [0.7, 0.8],
            "max_features": [0.7, 0.8]
        }
        
        print("\nTraining main model with regularization...")
        base_model = GradientBoostingRegressor(
            loss="huber",
            random_state=42,
            n_iter_no_change=10,
            validation_fraction=0.1
        )
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        print("\nBest Parameters:", grid_search.best_params_)
        
        # Train specialized models using best parameters
        print("\nTraining specialized models...")
        budget_idx = feature_names.index('log_budget')
        specialized_models = train_specialized_models(
            X_train, y_train, budget_idx,
            {'estimator': base_model, 'params': grid_search.best_params_}
        )
        
        # Generate predictions
        main_preds = best_model.predict(X_test)
        if specialized_models:
            spec_preds = weighted_prediction(X_test, specialized_models, budget_idx)
            final_preds = 0.7 * main_preds + 0.3 * spec_preds
        else:
            final_preds = main_preds
        
        # Calculate and print metrics
        train_metrics = calculate_metrics(y_train, best_model.predict(X_train))
        test_metrics = calculate_metrics(y_test, final_preds)
        
        # In the main execution section
        print("\nOptimized Model Performance:")
        print(f"Train R²: {train_metrics['r2']:.4f}")
        print(f"Test R²: {test_metrics['r2']:.4f}")

        # Handle potentially NaN values
        print(f"Test Weighted MAPE: {test_metrics.get('weighted_mape', 'N/A'):.2f}%")
        if not np.isnan(test_metrics.get('high_value_mape', np.nan)):
            print(f"High-value MAPE: {test_metrics['high_value_mape']:.2f}%")
        else:
            print("High-value MAPE: Insufficient samples")
        
        # Enhanced visualization
        plt.figure(figsize=(12, 8))
        plt.scatter(np.expm1(y_test), np.expm1(final_preds), alpha=0.3)
        plt.plot([1e4, 1e9], [1e4, 1e9], 'k--')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Actual Gross Profit (USD)")
        plt.ylabel("Predicted Gross Profit (USD)")
        plt.title("Actual vs Predicted Gross Profits")
        plt.grid(True)
        plt.savefig('results/predictions.png')
        plt.close()
        
        # Save models
        joblib.dump(best_model, 'saved_weights/main_model.pkl')
        if specialized_models:
            joblib.dump(specialized_models, 'saved_models/specialized_models.pkl')
        
        print("\nOptimization complete. Results saved.")

    except Exception as e:
        print(f"Main execution error: {str(e)}")
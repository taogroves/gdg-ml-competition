import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from prep_data import get_prepared_data
import matplotlib.pyplot as plt
# had to install libomp, and xgboost

def create_xgboost_model():
    # XGBoost parameters
    params = {
        'objective': 'reg:squarederror',  # for regression
        'learning_rate': 0.01,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 1000,
        'early_stopping_rounds': 50,
        'seed': 42
    }
    
    return xgb.XGBRegressor(**params)

def train_and_evaluate():
    # Get data
    features, target = get_prepared_data()
    
    # Convert to numpy arrays
    X = features.numpy()
    y = target.numpy().reshape(-1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = create_xgboost_model()
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100  # Print info every 100 boosting rounds
    )
    
    # Make predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)
    
    print("\nModel Performance:")
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Test R² Score: {test_r2:.4f}")
    
    # Plot feature importance
    plot_feature_importance(model)
    
    # Plot actual vs predicted
    plot_actual_vs_predicted(y_test, test_preds)
    
    return model

def plot_feature_importance(model):
    # Get feature importance
    importance = model.feature_importances_
    # Get feature names (you might need to modify this based on your feature names)
    features = [f"Feature_{i}" for i in range(len(importance))]
    
    # Sort by importance
    indices = np.argsort(importance)[::-1]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(indices[:20])), importance[indices[:20]])
    plt.xticks(range(len(indices[:20])), [features[i] for i in indices[:20]], rotation=45)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def plot_actual_vs_predicted(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')
    plt.close()

def predict_revenue(model, features):
    """
    Predict revenue for new movie features
    """
    prediction = model.predict(features.reshape(1, -1))
    return np.exp(prediction[0]) - 1  # Convert back from log scale

if __name__ == "__main__":
    # Train model
    model = train_and_evaluate()
    
    # Save model
    model.save_model('saved_weights/xgboost_model.json')
    print("\nModel saved as 'xgboost_model.json'")
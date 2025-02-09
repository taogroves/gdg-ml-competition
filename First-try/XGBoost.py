import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from features import prepare_features

# Add debug logging
def log_step(msg, df=None):
    print(msg)
    if df is not None:
        print(f"Shape: {df.shape}")
        print("Columns:", df.columns.tolist())
        print("---")

# Load and verify data
try:
    data_path = "data/IMDB top 1000.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    
    df = pd.read_csv(data_path)
    log_step("1. Data loaded successfully", df)

    # Preprocess data
    X, y = prepare_features(df)
    log_step("2. Features prepared", X)
    print("Target shape:", y.shape if y is not None else "None")

    if y is None:
        raise ValueError("Target variable not created during preprocessing")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    log_step("3. Data split complete")
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")

    param_grid = {
        "n_estimators": [100, 500],
        "max_depth": [3, 6],
        "learning_rate": [0.05, 0.1],
    }

    # Implementing GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb.XGBRegressor(objective="reg:squarederror", random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )

    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best Parameters:", best_params)
    print("Best R^2 Score:", best_score)

    best_model = xgb.XGBRegressor(
        objective="reg:squarederror", random_state=42, **best_params
    )
    best_model.fit(X_train, y_train)

    train_predictions = best_model.predict(X_train)
    test_predictions = best_model.predict(X_test)


    def calculate_metrics(y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        msle = mean_squared_log_error(y_true, y_pred)
        mape = np.mean(np.abs((np.exp(y_true) - np.exp(y_pred)) / np.exp(y_true))) * 100
        return r2, mse, msle, mape


    train_r2, train_mse, train_msle, train_mape = calculate_metrics(
        y_train, train_predictions
    )
    test_r2, test_mse, test_msle, test_mape = calculate_metrics(y_test, test_predictions)

    print(f"\nTraining Metrics:")
    print(f"R2 score: {train_r2:.4f}")
    print(f"MSE: {train_mse:.4f}")
    print(f"MLSE: {train_msle:.4f}")
    print(f"MAPE: {train_mape:.2f}%")

    print(f"\nTest Metrics:")
    print(f"R2 score: {test_r2:.4f}")
    print(f"MSE: {test_mse:.4f}")
    print(f"MSLE: {test_msle:.4f}")
    print(f"MAPE: {test_mape:.2f}%")

    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_train, train_predictions, color="blue", label="Train")
    plt.scatter(y_test, test_predictions, color="red", label="Test")
    plt.title("Actual vs Predicted Values")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.show()


    # Feature importance
    feature_importance = best_model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5

    plt.figure(figsize=(12, 8))
    plt.barh(pos, feature_importance[sorted_idx], align="center")
    plt.yticks(pos, X.columns[sorted_idx])
    plt.xlabel("Feature Importance")
    plt.title("Variable Importance")
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Error: {str(e)}")
    raise
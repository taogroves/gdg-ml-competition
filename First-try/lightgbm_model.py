import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from prep_data import get_prepared_data
import matplotlib.pyplot as plt
import pandas as pd
#had to install lightgbm

def create_lightgbm_model():
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.01,
        'num_leaves': 31,
        'max_depth': 6,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_estimators': 1000,
        'early_stopping_rounds': 50
    }
    
    return lgb.LGBMRegressor(**params)

def train_and_evaluate():
    try:
        # Get data
        features, target = get_prepared_data()
        
        # Get feature names
        feature_names = [
            'duration', 'rate', 'score', 'votes', 'year', 'budget', 'metascore',
            'genre_Film-Noir', 'genre_War', 'genre_Thriller', 'genre_Animation',
            'genre_Adventure', 'genre_Fantasy', 'genre_Family', 'genre_Crime',
            'genre_Sci-Fi', 'genre_Western', 'genre_Mystery', 'genre_Action',
            'genre_Horror', 'genre_Music', 'genre_Musical', 'genre_Sport',
            'genre_Biography', 'genre_Drama', 'genre_History', 'genre_Comedy',
            'genre_Romance', 'director_impact'
        ]
        
        # Convert to numpy arrays
        X = features.numpy()
        y = target.numpy().reshape(-1)
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("\nTraining model...")
        # Create and train model
        model = create_lightgbm_model()
        
        # Train with early stopping
        eval_set = [(X_test, y_test)]
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric=['rmse', 'mae'],
            verbose=100
        )
        
        # Make predictions
        print("\nMaking predictions...")
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        train_r2 = r2_score(y_train, train_preds)
        test_r2 = r2_score(y_test, test_preds)
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        
        print("\nModel Performance:")
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"Training R² Score: {train_r2:.4f}")
        print(f"Test R² Score: {test_r2:.4f}")
        
        # Plot feature importance
        plot_feature_importance(model, feature_names)
        
        # Plot actual vs predicted
        plot_actual_vs_predicted(y_test, test_preds)
        
        # Plot training history
        plot_training_history(model)
        
        return model
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

def plot_feature_importance(model, feature_names):
    try:
        # Get feature importance
        importance = model.feature_importances_
        
        # Create DataFrame for better visualization
        feature_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(feature_imp[:20])), feature_imp['Importance'][:20])
        plt.xticks(range(len(feature_imp[:20])), feature_imp['Feature'][:20], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
        # Print top 10 features
        print("\nTop 10 Most Important Features:")
        for idx, row in feature_imp[:10].iterrows():
            print(f"{row['Feature']}: {row['Importance']:.4f}")
            
    except Exception as e:
        print(f"Error in plotting feature importance: {str(e)}")

def plot_actual_vs_predicted(y_true, y_pred):
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.tight_layout()
        plt.savefig('actual_vs_predicted.png')
        plt.close()
    except Exception as e:
        print(f"Error in plotting actual vs predicted: {str(e)}")

def plot_training_history(model):
    try:
        evals_result = model.evals_result_
        
        plt.figure(figsize=(10, 6))
        plt.plot(evals_result['validation_0']['rmse'], label='RMSE')
        plt.xlabel('Iterations')
        plt.ylabel('RMSE')
        plt.title('Training History')
        plt.legend()
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
    except Exception as e:
        print(f"Error in plotting training history: {str(e)}")

if __name__ == "__main__":
    try:
        print("Starting model training...")
        model = train_and_evaluate()
        
        # Save model
        import joblib
        joblib.dump(model, 'saved_weights/lightgbm_model.pkl')
        print("\nModel saved as 'lightgbm_model.pkl'")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
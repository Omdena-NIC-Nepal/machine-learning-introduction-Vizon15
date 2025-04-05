import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_model(model_path):
    """Load the trained model."""
    return joblib.load(model_path)

def load_data(file_path, target_column):
    """Load dataset and split into features and target."""
    data = pd.read_csv(file_path)
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return X, y

def evaluate_metrics(y_true, y_pred):
    """Calculate evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

def plot_residuals(y_true, y_pred):
    """Plot residuals to check assumptions of linear regression."""
    residuals = y_true - y_pred
    
    # Residuals histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=30, kde=True, color="skyblue")
    plt.title("Residuals Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.axvline(0, color="red", linestyle="--")
    plt.show()
    
    # Residuals vs Predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color="red", linestyle="--")
    plt.title("Residuals vs Predictions")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.show()

def evaluate_model(model_path, data_path, target_column):
    """Complete model evaluation workflow."""
    # Load model and data
    model = load_model(model_path)
    X, y = load_data(data_path, target_column)
    
    # Split data (use only test set if pre-split)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate metrics
    mse, rmse, mae, r2 = evaluate_metrics(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R2): {r2:.2f}")
    
    # Plot residuals
    plot_residuals(y_test, y_pred)

# Main workflow
if __name__ == "__main__":
    model_file = "./models/linear_model.pkl"
    data_file = "./data/processed_boston_housing.csv"
    target_column = 'medv'
    
    evaluate_model(model_file, data_file, target_column)

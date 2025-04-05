import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def load_data(file_path):
    """Load the dataset."""
    return pd.read_csv(file_path)

def split_data(data, target_column, test_size=0.2, random_state=42):
    """Split the dataset into training and testing sets."""
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_linear_regression(X_train, y_train):
    """Train a linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate the model and return performance metrics."""
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    r2 = r2_score(y_test, test_predictions)
    
    return train_rmse, test_rmse, r2

def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning for Ridge regression."""
    ridge_model = Ridge()
    param_grid = {'alpha': [0.1, 1.0, 10.0]}
    grid_search = GridSearchCV(ridge_model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

def save_model(model, file_path):
    """Save the trained model to a file."""
    joblib.dump(model, file_path)

# Main Workflow
if __name__ == "__main__":
    # Step 1: Load Data
    data_file = "./data/processed_boston_housing.csv"
    data = load_data(data_file)

    # Step 2: Split Data
    target_column = 'medv'
    X_train, X_test, y_train, y_test = split_data(data, target_column)
    
    # Step 3: Train Model
    model = train_linear_regression(X_train, y_train)
    
    # Step 4: Evaluate Model
    train_rmse, test_rmse, r2 = evaluate_model(model, X_train, X_test, y_train, y_test)
    print("Model Coefficients:", model.coef_)
    print("Model Intercept:", model.intercept_)
    print(f"Training RMSE: {train_rmse:.2f}")
    print(f"Testing RMSE: {test_rmse:.2f}")
    print(f"R-squared: {r2:.2f}")
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    # Print first five predictions
    print("Predicted Prices:", y_pred[:5])

    # Step 5: Perform Hyperparameter Tuning (optional)
    best_model, best_params = hyperparameter_tuning(X_train, y_train)
    print("Best Hyperparameters:", best_params)

    # Step 6: Save Model
    save_model(best_model, "./models/linear_model.pkl")
    print("Model Saved Successfully")

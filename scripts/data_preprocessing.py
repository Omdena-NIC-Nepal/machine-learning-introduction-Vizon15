import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def load_data(file_path):
    """Load the dataset."""
    return pd.read_csv(file_path)

def handle_missing_values(data):
    """Handle missing values using SimpleImputer."""
    imputer = SimpleImputer(strategy="mean")  # Impute with column means
    numerical_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
    data[numerical_cols] = imputer.fit_transform(data[numerical_cols])
    return data

def handle_outliers(data):
    """Handle outliers using the IQR method."""
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    return data

def encode_categorical_variables(data):
    """Encode categorical variables using LabelEncoder."""
    if 'chas' in data.columns:  # Example: 'chas' is a binary categorical column
        le = LabelEncoder()
        data['chas'] = le.fit_transform(data['chas'])
    return data

def normalize_numerical_features(data, target_column):
    """Normalize numerical features using StandardScaler."""
    scaler = StandardScaler()
    numerical_features = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col]) and col != target_column]
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    return data

def split_data(data, target_column, test_size=0.2, random_state=42):
    """Split the dataset into training and testing sets."""
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def preprocess_data(file_path, target_column='medv'):
    """Complete data preprocessing workflow."""
    data = load_data(file_path)
    data = handle_missing_values(data)
    data = handle_outliers(data)
    data = encode_categorical_variables(data)
    data = normalize_numerical_features(data, target_column)
    X_train, X_test, y_train, y_test = split_data(data, target_column)
    
    return X_train, X_test, y_train, y_test

# File path for the dataset
file_path = "./data/BostonHousing.csv" 

# Run the preprocessing pipeline
X_train, X_test, y_train, y_test = preprocess_data(file_path)

print(f"Model Training Completed. Training set size: {X_train.shape}, Testing set size: {X_test.shape}")

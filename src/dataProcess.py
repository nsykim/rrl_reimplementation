import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file_path, column_names=None):
    """
    Load data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        Column_names (list): List of column names in case column names are not present in the CSV file.
    
    Returns:
        pd.DataFrame: Loaded data.
    """
    data = pd.read_csv(file_path, header=None, names=column_names)
    return data

def preprocess_data(data, numerical_features, categorical_features, target_feature=None):
    """
    Preprocess the data using scikit-learn's preprocessing utilities.
    
    Args:
        data (pd.DataFrame): Raw data.
        numerical_features (list): List of numerical feature names.
        categorical_features (list): List of categorical feature names.
        target_feature (str): Name of the target feature (optional).
    
    Returns:
        np.ndarray: Preprocessed data.
        np.ndarray: Target data (if target_feature is provided).
    """
    # Define the preprocessing for numerical features
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Define the preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Fit and transform the data
    features = data.drop(columns=[target_feature]) if target_feature else data
    preprocessed_data = preprocessor.fit_transform(features)
    
    if target_feature:
        target_data = data[target_feature]
        return preprocessed_data, target_data
    return preprocessed_data

def get_data_splits(data, target=None, test_size=0.2, stratify=False):
    """
    Split the data into training and testing sets.
    
    Args:
        data (np.ndarray): Preprocessed data.
        target (np.ndarray): Target data (optional).
        test_size (float): Proportion of the dataset to include in the test split.
        stratify (bool): Whether to stratify the split based on the target.
    
    Returns:
        tuple: Training and testing data.
    """
    if stratify and target is not None:
        train_data, test_data, train_target, test_target = train_test_split(
            data, target, test_size=test_size, stratify=target)
        return (train_data, train_target), (test_data, test_target)
    else:
        train_data, test_data = train_test_split(data, test_size=test_size)
        return train_data, test_data

def encode_labels(labels):
    """
    Encode labels using LabelEncoder.
    
    Args:
        labels (pd.Series): Labels to encode.
    
    Returns:
        np.ndarray: Encoded labels.
    """
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    return encoded_labels
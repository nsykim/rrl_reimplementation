import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from category_encoders import OneHotEncoder as CatEncOneHotEncoder  # Alternative one-hot encoder
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler


def read_info(info_path):
    """
    Reads information from a file and processes it into a list of tokens.

    Args:
        info_path (str): The path to the file containing the information.

    Returns:
        tuple: A tuple containing:
            - list: A list of lists, where each inner list contains tokens from each line of the file.
            - int: The integer value from the last token of the last line in the file.
    """
    with open(info_path) as f:
        f_list = []
        for line in f:
            tokens = line.strip().split()
            f_list.append(tokens)
    return f_list[:-1], int(f_list[-1][-1])


def read_csv(data_path, info_path, shuffle=False):
    """
    Reads a CSV file and processes it according to the provided information file.

    Args:
        data_path (str): The path to the CSV data file.
        info_path (str): The path to the information file that contains feature names and label position.
        shuffle (bool, optional): If True, shuffles the data. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - X_df (pd.DataFrame): DataFrame containing the features.
            - y_df (pd.DataFrame): DataFrame containing the labels.
            - f_df (pd.DataFrame): DataFrame containing the feature names.
            - label_pos (int): The position of the label column.
    """
    D = pd.read_csv(data_path, header=None)
    if shuffle:
        D = D.sample(frac=1, random_state=0).reset_index(drop=True)
    f_list, label_pos = read_info(info_path)
    f_df = pd.DataFrame(f_list)
    D.columns = f_df.iloc[:, 0]
    y_df = D.iloc[:, [label_pos]]
    X_df = D.drop(D.columns[label_pos], axis=1)
    f_df = f_df.drop(f_df.index[label_pos])
    return X_df, y_df, f_df, label_pos

class DBEncoder:
    """
    A class used to encode and preprocess data for machine learning models.
    Attributes
    ----------
    f_df : pd.DataFrame
        DataFrame containing feature information.
    discrete : bool, optional
        Whether to treat discrete features separately (default is False).
    y_one_hot : bool, optional
        Whether to one-hot encode the target variable (default is True).
    drop : str, optional
        Specifies a column to drop from one-hot encoding (default is 'first').
    Methods
    -------
    split_data(X_df)
        Splits the input DataFrame into discrete and continuous data.
    fit(X_df, y_df)
        Fits the encoder to the input features and target variable.
    transform(X_df, y_df, normalized=False, keep_stat=False)
        Transforms the input features and target variable using the fitted encoder.
    """

    def __init__(self, f_df, discrete=False, y_one_hot=True, drop='first'):
        """
        Initializes the data processing object.
        Parameters:
        f_df (pd.DataFrame): The input dataframe containing features.
        discrete (bool): Flag indicating if the features are discrete. Default is False.
        y_one_hot (bool): Flag indicating if the target variable should be one-hot encoded. Default is True.
        drop (str): Specifies a column to drop when using one-hot encoding. Default is 'first'.
        Attributes:
        f_df (pd.DataFrame): The input dataframe containing features.
        discrete (bool): Indicates if the features are discrete.
        y_one_hot (bool): Indicates if the target variable is one-hot encoded.
        label_enc (object): Encoder for the target variable.
        feature_enc (object): Encoder for the features.
        imp (SimpleImputer): Imputer for handling missing values.
        scaler (StandardScaler): Scaler for standardizing features.
        X_fname (str): Filename for the feature data.
        y_fname (str): Filename for the target data.
        discrete_flen (int): Length of discrete features.
        continuous_flen (int): Length of continuous features.
        mean (float): Mean of the features.
        std (float): Standard deviation of the features.
        """
        self.f_df = f_df
        self.discrete = discrete
        self.y_one_hot = y_one_hot
        # Use alternative encoders
        self.label_enc = CatEncOneHotEncoder() if y_one_hot else LabelEncoder()
        self.feature_enc = CatEncOneHotEncoder(cols=None, drop_invariant=True) if y_one_hot else OneHotEncoder(drop=drop)
        self.imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.scaler = StandardScaler()
        
        self.X_fname = None
        self.y_fname = None
        self.discrete_flen = None
        self.continuous_flen = None
        self.mean = None
        self.std = None

    def split_data(self, X_df):
        """
        Splits the input DataFrame into discrete and continuous data based on feature types.

        Args:
            X_df (pd.DataFrame): The input DataFrame containing the data to be split.

        Returns:
            tuple: A tuple containing two DataFrames:
                - discrete_data (pd.DataFrame): DataFrame containing the discrete features.
                - continuous_data (pd.DataFrame): DataFrame containing the continuous features.
                    Missing values in continuous features are replaced with NaN and converted to numeric type.
        """
        discrete_data = X_df[self.f_df[self.f_df[1] == 'discrete'].iloc[:, 0]]
        continuous_data = X_df[self.f_df[self.f_df[1] == 'continuous'].iloc[:, 0]]
        if not continuous_data.empty:
            continuous_data = continuous_data.replace(to_replace=r'.*\?.*', value=np.nan, regex=True)
            continuous_data = continuous_data.apply(pd.to_numeric, errors='coerce')
        return discrete_data, continuous_data

    def fit(self, X_df, y_df):
        """
        Fits the data processing pipeline to the provided feature and target dataframes.
        Parameters:
        -----------
        X_df : pandas.DataFrame
            The input features dataframe.
        y_df : pandas.DataFrame
            The target labels dataframe.
        Returns:
        --------
        None
        Notes:
        ------
        - Resets the index of both X_df and y_df.
        - Splits the input features into discrete and continuous data.
        - Fits the label encoder to the target labels.
        - Fits the imputer to the continuous data if available.
        - Fits the feature encoder to the discrete data if available.
        - Sets the feature names for both discrete and continuous data.
        - Updates the lengths of discrete and continuous feature sets.
        """
        X_df = X_df.reset_index(drop=True)
        y_df = y_df.reset_index(drop=True)
        discrete_data, continuous_data = self.split_data(X_df)
        
        if self.y_one_hot:
            self.label_enc.fit(y_df)
            self.y_fname = self.label_enc.get_feature_names() if hasattr(self.label_enc, 'get_feature_names') else y_df.columns
        else:
            self.label_enc.fit(y_df.values.ravel())
            self.y_fname = y_df.columns

        if not continuous_data.empty:
            self.imp.fit(continuous_data)
        
        if not discrete_data.empty:
            self.feature_enc.fit(discrete_data)
            feature_names = discrete_data.columns
            self.X_fname = list(self.feature_enc.get_feature_names()) if hasattr(self.feature_enc, 'get_feature_names') else feature_names.tolist()
            self.discrete_flen = len(self.X_fname)
            if not self.discrete:
                self.X_fname.extend(continuous_data.columns)
        else:
            self.X_fname = continuous_data.columns.tolist()
            self.discrete_flen = 0

        self.continuous_flen = continuous_data.shape[1]

    def transform(self, X_df, y_df, normalized=False, keep_stat=False):
        """
        Transforms the input dataframes by encoding labels, imputing missing values, 
        and optionally normalizing continuous features.
        Parameters:
        X_df (pd.DataFrame): DataFrame containing the feature data.
        y_df (pd.DataFrame): DataFrame containing the target labels.
        normalized (bool, optional): If True, normalizes the continuous features. Default is False.
        keep_stat (bool, optional): If True, keeps the mean and standard deviation of the continuous features 
                        for normalization. Default is False.
        Returns:
        tuple: A tuple containing:
            - np.ndarray: Transformed feature data.
            - np.ndarray: Transformed target labels.
        """
        X_df = X_df.reset_index(drop=True)
        y_df = y_df.reset_index(drop=True)
        discrete_data, continuous_data = self.split_data(X_df)
        
        # Label encoding for y
        y_encoded = self.label_enc.transform(y_df)
        if self.y_one_hot:
            y = y_encoded.toarray() if hasattr(y_encoded, 'toarray') else y_encoded
        else:
            y = y_encoded if isinstance(y_encoded, np.ndarray) else y_encoded.values.ravel()

        if not continuous_data.empty:
            continuous_data = pd.DataFrame(self.imp.transform(continuous_data), columns=continuous_data.columns)
            if normalized:
                if keep_stat:
                    self.mean = continuous_data.mean()
                    self.std = continuous_data.std()
                continuous_data = pd.DataFrame(self.scaler.fit_transform(continuous_data), columns=continuous_data.columns)

        if not discrete_data.empty:
            discrete_data_encoded = self.feature_enc.transform(discrete_data)
            discrete_data_encoded = pd.DataFrame(discrete_data_encoded, columns=self.X_fname[:self.discrete_flen])
            X_df = pd.concat([discrete_data_encoded, continuous_data], axis=1) if not self.discrete else discrete_data_encoded
        else:
            X_df = continuous_data

        return X_df.values, y
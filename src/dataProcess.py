import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer


def read_info(info_path):
    with open(info_path) as f:
        f_list = []
        for line in f:
            tokens = line.strip().split()
            f_list.append(tokens)
    return f_list[:-1], int(f_list[-1][-1])


def read_csv(data_path, info_path, shuffle=False):
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


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from category_encoders import OneHotEncoder as CatEncOneHotEncoder  # Alternative one-hot encoder

class DBEncoder:
    """Encoder used for data discretization and binarization."""

    def __init__(self, f_df, discrete=False, y_one_hot=True, drop='first'):
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
        discrete_data = X_df[self.f_df[self.f_df[1] == 'discrete'].iloc[:, 0]]
        continuous_data = X_df[self.f_df[self.f_df[1] == 'continuous'].iloc[:, 0]]
        if not continuous_data.empty:
            continuous_data = continuous_data.replace(to_replace=r'.*\?.*', value=np.nan, regex=True)
            continuous_data = continuous_data.apply(pd.to_numeric, errors='coerce')
        return discrete_data, continuous_data

    def fit(self, X_df, y_df):
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


# 
# class DBEncoder:
    # """Encoder used for data discretization and binarization."""
# 
    # def __init__(self, f_df, discrete=False, y_one_hot=True, drop='first'):
        # self.f_df = f_df
        # self.discrete = discrete
        # self.y_one_hot = y_one_hot
        # self.label_enc = preprocessing.OneHotEncoder(categories='auto') if y_one_hot else preprocessing.LabelEncoder()
        # self.feature_enc = preprocessing.OneHotEncoder(categories='auto', drop=drop)
        # self.imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        # self.X_fname = None
        # self.y_fname = None
        # self.discrete_flen = None
        # self.continuous_flen = None
        # self.mean = None
        # self.std = None
# 
    # def split_data(self, X_df):
        # discrete_data = X_df[self.f_df.loc[self.f_df[1] == 'discrete', 0]]
        # continuous_data = X_df[self.f_df.loc[self.f_df[1] == 'continuous', 0]]
        # if not continuous_data.empty:
            # continuous_data = continuous_data.replace(to_replace=r'.*\?.*', value=np.nan, regex=True)
            # continuous_data = continuous_data.astype(np.float)
        # return discrete_data, continuous_data
# 
    # def fit(self, X_df, y_df):
        # X_df = X_df.reset_index(drop=True)
        # y_df = y_df.reset_index(drop=True)
        # discrete_data, continuous_data = self.split_data(X_df)
        # self.label_enc.fit(y_df)
        # self.y_fname = list(self.label_enc.get_feature_names_out(y_df.columns)) if self.y_one_hot else y_df.columns
# 
        # if not continuous_data.empty:
            # self.imp.fit(continuous_data.values)
        # if not discrete_data.empty:
            # self.feature_enc.fit(discrete_data)
            # feature_names = discrete_data.columns
            # self.X_fname = list(self.feature_enc.get_feature_names_out(feature_names))
            # self.discrete_flen = len(self.X_fname)
            # if not self.discrete:
                # self.X_fname.extend(continuous_data.columns)
        # else:
            # self.X_fname = continuous_data.columns
            # self.discrete_flen = 0
        # self.continuous_flen = continuous_data.shape[1]
# 
    # def transform(self, X_df, y_df, normalized=False, keep_stat=False):
        # X_df = X_df.reset_index(drop=True)
        # y_df = y_df.reset_index(drop=True)
        # discrete_data, continuous_data = self.split_data(X_df)
        # y = self.label_enc.transform(y_df.values.reshape(-1, 1))
        # if self.y_one_hot:
            # y = y.toarray()
# 
        # if not continuous_data.empty:
            # continuous_data = pd.DataFrame(self.imp.transform(continuous_data.values),
                                        #    columns=continuous_data.columns)
            # if normalized:
                # if keep_stat:
                    # self.mean = continuous_data.mean()
                    # self.std = continuous_data.std()
                # continuous_data = (continuous_data - self.mean) / self.std
        # if not discrete_data.empty:
            # discrete_data = self.feature_enc.transform(discrete_data)
            # if not self.discrete:
                # X_df = pd.concat([pd.DataFrame(discrete_data.toarray()), continuous_data], axis=1)
            # else:
                # X_df = pd.DataFrame(discrete_data.toarray())
        # else:
            # X_df = continuous_data
        # return X_df.values, y
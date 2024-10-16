import unittest
import pandas as pd
import numpy as np
from src.dataProcess import load_data, preprocess_data, get_data_splits, encode_labels

class TestDataProcess(unittest.TestCase):

    def setUp(self):
        # Define column names for the dataset
        self.column_names = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
            'hours-per-week', 'native-country', 'income'
        ]
        
        # Load the actual dataset
        self.data = load_data('datasets/adults/adult.csv', column_names=self.column_names)
        
        # Define numerical and categorical features based on the dataset
        self.numerical_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        self.categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
        self.target_feature = 'income'

    def test_load_data(self):
        # Test loading data from a CSV file
        loaded_data = load_data('datasets/adults/adult.csv', column_names=self.column_names)
        self.assertEqual(loaded_data.shape, self.data.shape)
        self.assertListEqual(list(loaded_data.columns), self.column_names)

    def test_preprocess_data(self):
        # Test preprocessing data
        preprocessed_data, target_data = preprocess_data(
            self.data, self.numerical_features, self.categorical_features, self.target_feature)
        # Check if the preprocessed data has the expected number of columns
        num_categorical_columns = sum([len(self.data[col].unique()) for col in self.categorical_features])
        expected_columns = len(self.numerical_features) + num_categorical_columns
        self.assertEqual(preprocessed_data.shape[1], expected_columns)
        self.assertEqual(target_data.shape, (self.data.shape[0],))

    def test_preprocess_data_with_augmentation(self):
        # Test preprocessing data with augmentation
        preprocessed_data, target_data = preprocess_data(
            self.data, self.numerical_features, self.categorical_features, self.target_feature, augment=True, augmentation_factor=0.1)
        # Check if the preprocessed data has the expected number of columns
        num_categorical_columns = sum([len(self.data[col].unique()) for col in self.categorical_features])
        expected_columns = len(self.numerical_features) + num_categorical_columns
        self.assertEqual(preprocessed_data.shape[1], expected_columns)
        self.assertEqual(target_data.shape, (self.data.shape[0],))
        # Check if the data has been augmented
        self.assertFalse(np.array_equal(preprocessed_data, preprocess_data(
            self.data, self.numerical_features, self.categorical_features, self.target_feature)[0]))

    def test_get_data_splits(self):
        # Test splitting data into training and testing sets
        preprocessed_data, target_data = preprocess_data(
            self.data, self.numerical_features, self.categorical_features, self.target_feature)
        (train_data, train_target), (test_data, test_target) = get_data_splits(
            preprocessed_data, target_data, test_size=0.2, stratify=True)
        total_rows = self.data.shape[0]
        expected_train_rows = int(0.8 * total_rows)
        expected_test_rows = total_rows - expected_train_rows
        self.assertEqual(train_data.shape[0], expected_train_rows)  # 80% of rows
        self.assertEqual(test_data.shape[0], expected_test_rows)    # 20% of rows
        self.assertEqual(train_target.shape[0], expected_train_rows)
        self.assertEqual(test_target.shape[0], expected_test_rows)

    def test_encode_labels(self):
        # Test encoding labels
        labels = self.data['workclass']
        encoded_labels = encode_labels(labels)
        self.assertEqual(len(encoded_labels), len(labels))
        self.assertTrue((encoded_labels == encode_labels(labels)).all())

    def test_preprocess_data_no_target(self):
        # Test preprocessing data without a target feature
        preprocessed_data = preprocess_data(
            self.data, self.numerical_features, self.categorical_features)
        num_categorical_columns = sum([len(self.data[col].unique()) for col in self.categorical_features])
        expected_columns = len(self.numerical_features) + num_categorical_columns
        self.assertEqual(preprocessed_data.shape[1], expected_columns)

    def test_get_data_splits_no_target(self):
        # Test splitting data into training and testing sets without a target
        preprocessed_data = preprocess_data(
            self.data, self.numerical_features, self.categorical_features)
        train_data, test_data = get_data_splits(preprocessed_data, test_size=0.2)
        total_rows = self.data.shape[0]
        expected_train_rows = int(0.8 * total_rows)
        expected_test_rows = total_rows - expected_train_rows
        self.assertEqual(train_data.shape[0], expected_train_rows)  # 80% of rows
        self.assertEqual(test_data.shape[0], expected_test_rows)    # 20% of rows

    def test_encode_labels_empty(self):
        # Test encoding an empty label series
        labels = pd.Series([], dtype="str")
        encoded_labels = encode_labels(labels)
        self.assertEqual(len(encoded_labels), 0)

    def test_preprocess_data_with_missing_values(self):
        # Test preprocessing data with missing values
        data_with_missing = self.data.copy()
        data_with_missing.loc[0, 'age'] = np.nan
        preprocessed_data, target_data = preprocess_data(
            data_with_missing, self.numerical_features, self.categorical_features, self.target_feature)
        self.assertEqual(preprocessed_data.shape[0], self.data.shape[0])
        self.assertEqual(target_data.shape[0], self.data.shape[0])

    def test_load_data_no_columns(self):
        # Test loading data without specifying column names
        data = load_data('datasets/adults/adult.csv')
        self.assertEqual(data.shape[1], len(self.column_names))

    def test_preprocess_data_different_parts(self):
        # Test preprocessing different parts of the dataset
        data_part = self.data.iloc[:100]  # Use the first 100 rows
        preprocessed_data, target_data = preprocess_data(
            data_part, self.numerical_features, self.categorical_features, self.target_feature)
        num_categorical_columns = sum([len(data_part[col].unique()) for col in self.categorical_features])
        expected_columns = len(self.numerical_features) + num_categorical_columns
        self.assertEqual(preprocessed_data.shape[1], expected_columns)
        self.assertEqual(target_data.shape, (data_part.shape[0],))

    def test_preprocess_data_with_augmentation_different_parts(self):
        # Test preprocessing different parts of the dataset with augmentation
        data_part = self.data.iloc[:100]  # Use the first 100 rows
        preprocessed_data, target_data = preprocess_data(
            data_part, self.numerical_features, self.categorical_features, self.target_feature, augment=True, augmentation_factor=0.1)
        num_categorical_columns = sum([len(data_part[col].unique()) for col in self.categorical_features])
        expected_columns = len(self.numerical_features) + num_categorical_columns
        self.assertEqual(preprocessed_data.shape[1], expected_columns)
        self.assertEqual(target_data.shape, (data_part.shape[0],))
        # Check if the data has been augmented
        self.assertFalse(np.array_equal(preprocessed_data, preprocess_data(
            data_part, self.numerical_features, self.categorical_features, self.target_feature)[0]))

if __name__ == '__main__':
    unittest.main()
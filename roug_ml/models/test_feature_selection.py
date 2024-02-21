import unittest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import unittest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import numpy as np

import unittest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from roug_ml.models.feature_selection import SelectKBestOneHotSelector, ModelBasedOneHotSelector, \
    TreeFeatureSelector, RFEFeatureSelector, VarianceThresholdSelector
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


class TestSelectKBestOneHotSelector(unittest.TestCase):
    def setUp(self):
        # Setup for each test
        self.X = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]
        ])
        self.y = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        self.selector = SelectKBest(chi2, k=2)

    def test_fit_transform(self):
        # Test fitting and transforming functionality
        oh_selector = SelectKBestOneHotSelector(self.selector)
        oh_selector.fit(self.X, self.y)
        X_transformed = oh_selector.transform(self.X)

        self.assertEqual(X_transformed.shape[1], 2, "The transform method should select the two best features.")

    def test_get_support_mask(self):
        # Test getting support as a mask
        oh_selector = SelectKBestOneHotSelector(self.selector)
        oh_selector.fit(self.X, self.y)
        support_mask = oh_selector.get_support()

        self.assertIsInstance(support_mask, np.ndarray, "The get_support method should return an ndarray.")
        self.assertEqual(support_mask.sum(), 2, "There should be two features selected.")

    def test_get_support_indices(self):
        # Test getting support as indices
        oh_selector = SelectKBestOneHotSelector(self.selector)
        oh_selector.fit(self.X, self.y)
        support_indices = oh_selector.get_support(indices=True)

        self.assertIsInstance(support_indices, np.ndarray, "The get_support method should return an ndarray when indices=True.")
        self.assertEqual(len(support_indices), 2, "There should be two features selected.")


class TestModelBasedOneHotSelector(unittest.TestCase):
    def setUp(self):
        # Setup for each test
        self.X = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]
        ])
        self.y = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        self.estimator = RandomForestClassifier(n_estimators=10, random_state=42)

    def test_fit_transform(self):
        # Test fitting and transforming functionality
        oh_selector = ModelBasedOneHotSelector(self.estimator)
        oh_selector.fit(self.X, self.y)
        X_transformed = oh_selector.transform(self.X)

        # The number of selected features can vary depending on the estimator and data
        self.assertTrue(X_transformed.shape[1] <= self.X.shape[1], "The transform method should not increase the number of features.")

    def test_get_support_mask(self):
        # Test getting support as a mask
        oh_selector = ModelBasedOneHotSelector(self.estimator)
        oh_selector.fit(self.X, self.y)
        support_mask = oh_selector.get_support()

        self.assertIsInstance(support_mask, np.ndarray, "The get_support method should return an ndarray.")
        self.assertTrue(np.any(support_mask), "At least one feature should be selected.")

    def test_get_support_indices(self):
        # Test getting support as indices
        oh_selector = ModelBasedOneHotSelector(self.estimator)
        oh_selector.fit(self.X, self.y)
        support_indices = oh_selector.get_support(indices=True)

        self.assertIsInstance(support_indices, np.ndarray, "The get_support method should return an ndarray when indices=True.")
        self.assertTrue(len(support_indices) > 0, "There should be at least one feature selected.")


class TestTreeFeatureSelector(unittest.TestCase):
    def setUp(self):
        # Initial setup for X and y, with y being the original labels
        self.X = np.array([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15]
        ])
        self.y = np.array([0, 1, 2, 3])  # Original labels
        # Convert to one-hot encoding for use in tests where needed
        self.y_one_hot = np.eye(np.max(self.y) + 1)[self.y]

    def test_fit_transform_with_threshold(self):
        # Using y_one_hot instead of y for fitting
        selector = TreeFeatureSelector(n_estimators=10, max_depth=3, threshold=0.1)
        selector.fit(self.X, self.y_one_hot)  # Use one-hot encoded labels here
        X_transformed = selector.transform(self.X)
        self.assertTrue(X_transformed.shape[1] <= self.X.shape[1],
                        "Transformed dataset should have fewer features.")

    def test_fit_transform_with_default_threshold(self):
        # Using y_one_hot instead of y for fitting
        selector = TreeFeatureSelector(n_estimators=10, max_depth=3)
        selector.fit(self.X, self.y_one_hot)  # Use one-hot encoded labels here
        X_transformed = selector.transform(self.X)
        self.assertTrue(X_transformed.shape[1] <= self.X.shape[1],
                        "Transformed dataset should have fewer or the same number of features.")

    def test_feature_importance_threshold(self):
        # Using y_one_hot instead of y for fitting
        selector = TreeFeatureSelector(n_estimators=10, max_depth=3)
        selector.fit(self.X, self.y_one_hot)  # Use one-hot encoded labels here
        calculated_threshold = np.mean(selector.estimator.feature_importances_)
        self.assertAlmostEqual(selector.threshold, calculated_threshold,
                               "Threshold should be set to the mean of the feature importances by default.")

    def test_selection_indices(self):
        # Using y_one_hot instead of y for fitting
        selector = TreeFeatureSelector(n_estimators=10, max_depth=3, threshold=0.1)
        selector.fit(self.X, self.y_one_hot)  # Use one-hot encoded labels here
        expected_indices = [i for i, importance in enumerate(selector.estimator.feature_importances_) if importance > selector.threshold]
        self.assertEqual(list(selector.indices), expected_indices,
                         "Selected feature indices should match those expected based on the threshold.")


class TestRFEFeatureSelector(unittest.TestCase):
    def setUp(self):
        # Generate a simple classification dataset
        self.X, self.y = make_classification(n_samples=100, n_features=20, n_informative=2,
                                             n_redundant=2, n_classes=2, random_state=42)
        # Convert labels to one-hot encoding
        self.y_one_hot = np.eye(np.max(self.y) + 1)[self.y]

    def test_fit_transform(self):
        # Initialize RFEFeatureSelector with a simple estimator
        estimator = LogisticRegression(solver='liblinear')
        selector = RFEFeatureSelector(estimator=estimator, n_features_to_select=5)

        # Fit and transform using the one-hot encoded labels
        selector.fit(self.X, self.y_one_hot)
        X_transformed = selector.transform(self.X)

        # Check if the number of features in the transformed data matches n_features_to_select
        self.assertEqual(X_transformed.shape[1], 5, "Transformed dataset should have 5 features.")

    def test_fit_attributes(self):
        # Testing if the selector correctly fits and sets attributes
        estimator = LogisticRegression(solver='liblinear')
        selector = RFEFeatureSelector(estimator=estimator, n_features_to_select=5)

        selector.fit(self.X, self.y_one_hot)

        # Check if the selector has been fitted and has the attribute ranking_
        self.assertTrue(hasattr(selector.selector, 'ranking_'),
                        "Selector should have attribute 'ranking_' after fitting.")
        self.assertTrue(hasattr(selector.selector, 'support_'),
                        "Selector should have attribute 'support_' after fitting.")


class TestVarianceThresholdSelector(unittest.TestCase):
    def setUp(self):
        # Create a dataset with some features having zero variance and some with non-zero variance
        self.X = np.array([[0, 2, 1, 3.7],
                           [0, 2, 1, 3.2],
                           [0, 2, 1.01, 3.1],
                           [0, 2, 1, 3.8]])
        # No need for y in this case, as VarianceThreshold does not use it

    def test_remove_zero_variance(self):
        selector = VarianceThresholdSelector(threshold=0.0)
        selector.fit(self.X)
        X_transformed = selector.transform(self.X)
        self.assertEqual(X_transformed.shape[1], 2, "Transformed dataset should have 2 feature.")

    def test_custom_threshold(self):
        selector = VarianceThresholdSelector(threshold=0.08)
        selector.fit(self.X)
        X_transformed = selector.transform(self.X)
        expected_feature_count = 1
        self.assertEqual(X_transformed.shape[1], expected_feature_count,
                         "Transformed dataset should have {} features after applying a custom threshold.".format(
                             expected_feature_count))


if __name__ == '__main__':
    unittest.main()

import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import tempfile
from joblib import dump, load
from sklearn.exceptions import ConvergenceWarning
import warnings
from io import StringIO
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from textblob import TextBlob
import logging

# Mock genai to avoid API calls
from unittest.mock import patch, Mock
genai.configure = Mock()
genai.GenerativeModel = Mock(return_value=Mock(generate_content=Mock(return_value=Mock(text="{}"))))

# Import the functions from app.py
from app import compute_eda, train_model, train_clustering

class TestAppFunctions(unittest.TestCase):

    def setUp(self):
        # Set up logging for debugging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def test_compute_eda_basic(self):
        data = {'x': [1, 2, 3], 'y': [4, 5, 6]}
        df = pd.DataFrame(data)
        eda_data = compute_eda(df)
        self.assertEqual(eda_data['shape'], (3, 2))
        self.assertIn('x', eda_data['dtypes'])
        self.assertEqual(eda_data['dtypes']['x'], 'int64')
        self.assertEqual(eda_data['missing']['x'], 0)

    def test_compute_eda_empty(self):
        df = pd.DataFrame()
        try:
            eda_data = compute_eda(df)
            self.assertEqual(eda_data['shape'], (0, 0))
            self.assertEqual(eda_data['describe'], {})
            self.assertEqual(eda_data['dtypes'], {})
            self.assertEqual(eda_data['missing'], {})
        except Exception as e:
            self.logger.error(f"compute_eda_empty failed with exception: {e}")
            self.fail(f"compute_eda raised an unexpected exception: {e}")

    def test_compute_eda_non_numeric(self):
        data = {'category': ['a', 'b', 'c']}
        df = pd.DataFrame(data)
        eda_data = compute_eda(df)
        self.assertEqual(eda_data['shape'], (3, 1))
        self.assertIn('category', eda_data['dtypes'])
        self.assertEqual(eda_data['dtypes']['category'], 'object')

    def test_train_model_linear_regression(self):
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
        mdl, metric_value, cv_mean, cv_std, y_test, y_pred = train_model(X, y, "linear-regression", 0.01, 50)
        self.assertIsNotNone(mdl)
        self.assertLess(metric_value, 1e-5)
        self.assertLess(cv_std, 1e-4)

    # In test_app.py

    def test_train_model_logistic_regression(self):
        np.random.seed(42)  # Ensure reproducible split
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]])
        y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])  # More balanced data

        # FIX: Use a standard tolerance (1e-4) instead of the loose 0.01.
        # This allows the model to converge properly to a good solution.
        # The call signature is: train_model(X, y, model_type, learning_rate, epochs)
        mdl, metric_value, cv_mean, cv_std, y_test, y_pred = train_model(X, y, "logistic-regression", 1e-4, 100)

        self.assertIsNotNone(mdl)
        self.assertGreaterEqual(metric_value, 1)
        self.assertGreaterEqual(cv_mean, 1)

    def test_train_model_with_tuning(self):
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
        mdl, metric_value, cv_mean, cv_std, y_test, y_pred = train_model(X, y, "rf-regression", 0.01, 50, tuning=True)
        self.assertIsNotNone(mdl)
        self.assertLess(metric_value, 1)

    def test_train_model_with_feature_eng(self):
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        y = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
        mdl, metric_value, cv_mean, cv_std, y_test, y_pred = train_model(X, y, "linear-regression", 0.01, 50, feature_eng=True)
        self.assertIsNotNone(mdl)
        self.assertLess(metric_value, 1e-5)

    def test_train_model_insufficient_data(self):
        X = np.array([[1]])
        y = np.array([1])
        try:
            mdl, metric_value, cv_mean, cv_std, y_test, y_pred = train_model(X, y, "linear-regression", 0.01, 50)
            self.assertIsNone(mdl)
            self.assertIsNone(metric_value)
            self.assertIsNone(cv_mean)
            self.assertIsNone(cv_std)
            self.assertIsNone(y_test)
            self.assertIsNone(y_pred)
        except Exception as e:
            self.logger.error(f"train_model_insufficient_data failed with exception: {e}")
            self.fail(f"train_model raised an unexpected exception: {e}")

    def test_train_clustering(self):
        data_scaled = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        kmeans, labels, inertia, data_reduced = train_clustering(data_scaled, 2)
        self.assertIsNotNone(kmeans)
        self.assertEqual(len(labels), 6)
        self.assertGreater(inertia, 0)
        self.assertEqual(data_reduced.shape[1], 2)

    def test_train_clustering_tsne(self):
        data_scaled = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]])
        try:
            kmeans, labels, inertia, data_reduced = train_clustering(data_scaled, 2, reduction_method='t-SNE')
            self.assertIsNotNone(kmeans)
            self.assertEqual(len(labels), 8)
            self.assertGreater(inertia, 0)
            self.assertEqual(data_reduced.shape[1], 2)
        except Exception as e:
            self.logger.error(f"train_clustering_tsne failed with exception: {e}")
            self.fail(f"train_clustering raised an unexpected exception: {e}")

if __name__ == '__main__':
    unittest.main()
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
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

def truncate_value(value, max_length=150):
    """Truncates a value if it's a string longer than max_length."""
    if isinstance(value, str) and len(value) > max_length:
        return value[:max_length] + '...'
    return value

def prepare_data_for_prompt(df_snippet):
    """
    Prepares a DataFrame snippet for an API prompt by converting it to a
    dictionary and truncating long string values.
    """
    records = df_snippet.to_dict('records')
    truncated_records = [
        {key: truncate_value(val) for key, val in record.items()}
        for record in records
    ]
    return truncated_records

# --- Security and Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Setup ---
warnings.filterwarnings("ignore", category=ConvergenceWarning)
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    pass
else:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')

# --- Cache Configuration ---
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Prompt Templates ---
RECOMMENDATION_PROMPT_TEMPLATE = """
You are an expert data scientist providing a recommendation. A user has uploaded a dataset.
The column headers are: {headers}.
The first 5 rows of sample data are: {sample_data}.

Your task is to analyze these headers and sample data to suggest a machine learning task.
1. First, determine if the user's goal is likely prediction (supervised learning) or pattern discovery (unsupervised learning). If there is a clear target column with a strong relationship to other columns, suggest supervised learning. If there isn't a clear target, or the user might be interested in grouping their data, suggest unsupervised learning.
2. Based on your decision, recommend a model from this list: ['linear-regression', 'logistic-regression', 'rf-classification', 'rf-regression', 'kmeans-clustering'].
   - For 'linear-regression' or 'rf-regression', the target (Y) must be a continuous number.
   - For 'logistic-regression' or 'rf-classification', the target (Y) must be binary (like 0/1, true/false).
   - Recommend a Random Forest ('rf-') model over a simple one if the data seems complex or has many features.
   - Recommend 'kmeans-clustering' if you chose unsupervised learning.
3. Identify the best column(s) for the task.
   - For supervised models, identify the best input feature (X) and target (Y).
   - For 'kmeans-clustering', identify the two best numerical features to use for clustering and plotting. The first feature will be 'recommended_x' and the second will be 'recommended_y'.
4. Provide a brief, one-sentence explanation for your choice.

IMPORTANT: Respond ONLY with a valid JSON object in the following format.
{{
  "recommended_x": "header_name_for_x",
  "recommended_y": "header_name_for_y",
  "recommended_model": "model_name_from_list",
  "explanation": "Your one-sentence explanation here."
}}
"""

INTERPRETATION_PROMPT_TEMPLATE = """
You are a friendly and helpful data science explainer. A user has just trained a machine learning model and needs help understanding the result.
Here is the information about their model:
- Model Type: {model_type}
- Feature(s) Used: {x_column}, {y_column}
- Performance Metric: {metric_name}
- Metric Value: {metric_value:.4f}
Your task is to provide a simple, encouraging, one-paragraph explanation of what this result means. Your tone should be accessible to a complete beginner.
- If the model is 'Linear Regression' or 'Random Forest Regression', explain what Mean Squared Error (MSE) is (a measure of error, where lower is better).
- If the model is 'Logistic Regression' or 'Random Forest Classification', explain what Accuracy is (the percentage of correct predictions, where higher is better). Mention that Random Forest is a more powerful model.
- If the model is 'K-Means Clustering', the metric value is the inertia (within-cluster sum of squares, lower is better). Explain what clustering is (finding natural groups in data) and what a lower inertia suggests about cluster quality.
IMPORTANT: Respond ONLY with a valid JSON object in the following format.
{{
  "interpretation": "Your one-paragraph explanation here."
}}
"""


# --- Cache EDA Computation with Disk Backup ---
@st.cache_data
def compute_eda(df, sample_size=1000):
    """
    Computes Exploratory Data Analysis statistics for a DataFrame.
    """
    if df.empty:
        return {
            'shape': (0, 0),
            'describe': {},
            'dtypes': {},
            'missing': {}
        }


    sampled_df = df.sample(n=min(sample_size, len(df)), random_state=42) if len(df) > sample_size else df
    return {
        'shape': df.shape,
        'describe': sampled_df.describe(include='all').to_dict(),
        'dtypes': sampled_df.dtypes.astype(str).to_dict(),
        'missing': sampled_df.isnull().sum().to_dict()
    }


# --- Cache Model Training ---
@st.cache_data
def train_model(X, y, model_type, learning_rate, epochs, random_state=42, tuning=False, feature_eng=False):
    """
    Trains a supervised learning model using a robust, unified pipeline approach.
    """
    if X.shape[0] < 5:
        return None, None, None, None, None, None, None


    # Step 1: Define the sequence of steps for the pipeline
    pipeline_steps = []
    pipeline_steps.append(('scaler', StandardScaler()))

    if feature_eng:
        pipeline_steps.append(('poly_features', PolynomialFeatures(degree=2, include_bias=False)))

    # Define the base model estimator
    if model_type == "linear-regression":
        estimator = LinearRegression()
    elif model_type == "logistic-regression":
        estimator = LogisticRegression(solver='liblinear', max_iter=epochs, tol=learning_rate,
                                       random_state=random_state)
    elif model_type == "rf-regression":
        estimator = RandomForestRegressor(random_state=random_state)
    elif model_type == "rf-classification":
        estimator = RandomForestClassifier(random_state=random_state)
    elif model_type == "svm-regression":
        estimator = SVR()
    elif model_type == "svm-classification":
        estimator = SVC(random_state=random_state)
    elif model_type == "nn-regression":
        estimator = MLPRegressor(hidden_layer_sizes=(100,), max_iter=epochs, learning_rate_init=learning_rate,
                                 random_state=random_state)
    elif model_type == "nn-classification":
        estimator = MLPClassifier(hidden_layer_sizes=(100,), max_iter=epochs, learning_rate_init=learning_rate,
                                  random_state=random_state)
    else:
        return None, None, None, None, None, None, None

    pipeline_steps.append(('model', estimator))

    pipeline = Pipeline(steps=pipeline_steps)

    if tuning and model_type in ["rf-regression", "rf-classification", "svm-regression", "svm-classification"]:
        if model_type.startswith("rf"):
            param_grid = {'model__n_estimators': [50, 100], 'model__max_depth': [5, 10]}
        else:  # SVM
            param_grid = {'model__C': [0.1, 1], 'model__kernel': ['rbf', 'linear']}
        final_model = GridSearchCV(pipeline, param_grid, cv=3)
    else:
        if model_type.startswith("rf"):
            pipeline.set_params(model__n_estimators=100, model__max_depth=10)
        elif model_type.startswith("svm"):
            pipeline.set_params(model__kernel='rbf')
        final_model = pipeline

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)
    metric_value = accuracy_score(y_test, y_pred) if not model_type.endswith("regression") else mean_squared_error(
        y_test, y_pred)


    # Step 5: Perform cross-validation if possible
    cv_mean, cv_std = np.nan, np.nan  # Default values
    n_folds = 5

    can_run_cv = True
    if not model_type.endswith("regression"):
        min_class_count = pd.Series(y).value_counts().min()
        if min_class_count < n_folds:
            can_run_cv = False
            logger.warning(
                f"Skipping cross-validation because the smallest class has only {min_class_count} members, which is less than n_splits={n_folds}.")

    if can_run_cv:
        scoring = 'accuracy' if not model_type.endswith("regression") else 'neg_mean_squared_error'
        try:
            cv_scores = cross_val_score(final_model, X, y, cv=n_folds, scoring=scoring, n_jobs=-1)
            cv_mean = cv_scores.mean() if not model_type.endswith("regression") else -cv_scores.mean()
            cv_std = cv_scores.std()
        except ValueError as e:
            logger.error(f"Cross-validation failed unexpectedly: {e}")

    return final_model, metric_value, cv_mean, cv_std, X_test, y_test, y_pred


# --- Cache Clustering ---
@st.cache_data
def train_clustering(data_scaled, k, reduction_method='PCA', random_state=42):
    """
    Performs K-Means clustering with optional dimensionality reduction.
    """
    if reduction_method == 'PCA':
        pca = PCA(n_components=2)
        data_reduced = pca.fit_transform(data_scaled)
    elif reduction_method == 't-SNE':
        # Perplexity must be less than n_samples. The scikit-learn default is 30.
        n_samples = data_scaled.shape[0]

        # Perplexity must be > 0, which implies n_samples > 1.
        if n_samples > 1:
            perplexity_value = min(30, n_samples - 1)
            # Use recommended params for newer scikit-learn versions for stability
            tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=random_state, init='pca',
                        learning_rate='auto')
            data_reduced = tsne.fit_transform(data_scaled)
        else:
            # Fallback for n_samples=1 where t-SNE is not possible.
            pca = PCA(n_components=2)
            data_reduced = pca.fit_transform(data_scaled)  # Use PCA as a robust alternative
    else:  # Fallback for unknown methods
        pca = PCA(n_components=2)
        data_reduced = pca.fit_transform(data_scaled)

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(data_reduced)
    inertia = kmeans.inertia_
    return kmeans, labels, inertia, data_reduced


# --- Initialize Session State ---
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'headers' not in st.session_state:
    st.session_state['headers'] = []
if 'rec' not in st.session_state:
    st.session_state['rec'] = None
if 'x_col' not in st.session_state:
    st.session_state['x_col'] = []
if 'y_col' not in st.session_state:
    st.session_state['y_col'] = None
if 'feature1' not in st.session_state:
    st.session_state['feature1'] = None
if 'feature2' not in st.session_state:
    st.session_state['feature2'] = None
if 'model_type_sup' not in st.session_state:
    st.session_state['model_type_sup'] = 'linear-regression'
if 'k' not in st.session_state:
    st.session_state['k'] = 3

# --- Main App ---
st.title("Data Science Playground")
st.write("An intelligent environment for data exploration and machine learning. Please upload a CSV file to start.")

uploaded_file = st.file_uploader("Upload a CSV file to begin:", type="csv", key="file_uploader")
if uploaded_file is not None:
    try:
        # Security: Limit file size (e.g., 100MB)
        if uploaded_file.size > 100 * 1024 * 1024:
            st.error("File too large. Please upload a file smaller than 100MB.")
            st.stop()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            # Security: Sanitize file path
            safe_path = tmp_file.name.replace("../", "").replace("..\\", "")
            df = pd.read_csv(safe_path, chunksize=10000)
            df = pd.concat(chunk for chunk in df)
        st.session_state['df'] = df
        st.session_state['headers'] = df.columns.tolist()
        os.unlink(safe_path)
    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}. Please ensure the file is a valid CSV without malicious content.")
        logger.error(f"CSV reading error: {str(e)}")
        st.session_state['df'] = None
        st.session_state['headers'] = []


if st.session_state['df'] is not None:
    df = st.session_state['df']
    headers = st.session_state['headers']

    tab1, tab2 = st.tabs(["1. Data Explorer", "2. Model Training"])

    with tab1:
        st.header("Exploratory Data Analysis (EDA)")
        progress_bar = st.progress(0)
        eda_data = compute_eda(df)
        progress_bar.progress(100)
        st.write(f"Dataset Shape: {eda_data['shape'][0]} rows × {eda_data['shape'][1]} columns")
        st.subheader("Statistical Summary (Sampled)")
        st.dataframe(pd.DataFrame(eda_data['describe']))
        st.subheader("Data Types (Sampled)")
        st.dataframe(pd.Series(eda_data['dtypes'], name='dtype'))
        st.subheader("Missing Values (Sampled)")
        st.dataframe(pd.Series(eda_data['missing'], name='missing'))
        st.subheader("Data Preview (Paginated)")
        page_size = 10
        page = st.number_input("Page", min_value=1, value=1, step=1, key="page")
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        st.dataframe(df.iloc[start_idx:end_idx])
        st.subheader("Column Visualizations")
        col_vis = st.selectbox("Select column to visualize", headers, key="col_vis")
        sampled_data = df[col_vis].sample(n=min(1000, len(df)), random_state=42)
        fig = px.histogram(sampled_data, x=col_vis, nbins=30) if pd.api.types.is_numeric_dtype(df[col_vis]) else px.histogram(sampled_data, y=col_vis)
        fig.update_layout(title=f"Distribution of {col_vis} (Sampled)", xaxis_title=col_vis, yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("Download Chart", fig.to_json(), f"{col_vis}_chart.json", "application/json")

    with tab2:
        st.header("AI-Powered Setup")
        if st.button("Get AI Recommendation", key="ai_rec"):
            with st.spinner("Analyzing..."):
                try:
                    sample_data_for_prompt = prepare_data_for_prompt(df.head(5))

                    prompt = RECOMMENDATION_PROMPT_TEMPLATE.format(headers=headers, sample_data=sample_data_for_prompt)

                    response = model.generate_content(prompt)
                    cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
                    rec = json.loads(cleaned_text)
                    st.session_state['rec'] = rec
                    if rec['recommended_model'] == 'kmeans-clustering':
                        st.session_state['feature1'] = rec['recommended_x']
                        st.session_state['feature2'] = rec['recommended_y']
                    else:
                        st.session_state['x_col'] = [rec['recommended_x']]
                        st.session_state['y_col'] = rec['recommended_y']
                        st.session_state['model_type_sup'] = rec['recommended_model']
                    st.success(f"AI Suggestion: {rec['explanation']}")
                except Exception as e:
                    st.error(f"Error getting AI recommendation: {e}")

        learning_type = st.radio("Learning Type", ("Supervised", "Unsupervised"), horizontal=True, key="learning_type")

        if learning_type == "Supervised" and headers:
            st.subheader("Model Configuration")
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.multiselect("Input Features (X):", headers, default=st.session_state['x_col'], key="x_col")
            with col2:
                y_col = st.selectbox("Target (Y):", headers, index=headers.index(st.session_state['y_col']) if st.session_state['y_col'] in headers else 0, key="y_col")
            model_type_sup = st.selectbox("Model Type:", ("linear-regression", "logistic-regression", "rf-regression", "rf-classification", "svm-regression", "svm-classification", "nn-regression", "nn-classification"), index=0, key="model_type_sup")
            learning_rate = st.slider("Learning Rate:", 0.001, 0.1, value=0.01, step=0.001, key="learning_rate")
            epochs = st.slider("Epochs:", 10, 200, value=50, step=1, key="epochs")
            tuning = st.checkbox("Enable Hyperparameter Tuning", value=False, key="tuning")
            feature_eng = st.checkbox("Enable Feature Engineering (Polynomial + One-Hot)", value=False, key="feature_eng")

        elif learning_type == "Unsupervised" and headers:
            st.subheader("Clustering Configuration")
            numeric_cols = [col for col in headers if pd.api.types.is_numeric_dtype(df[col])]
            col1, col2 = st.columns(2)
            with col1:
                feature1 = st.selectbox("Feature 1:", numeric_cols, index=numeric_cols.index(st.session_state['feature1']) if st.session_state['feature1'] in numeric_cols else 0, key="feature1")
            with col2:
                feature2 = st.selectbox("Feature 2:", numeric_cols, index=numeric_cols.index(st.session_state['feature2']) if st.session_state['feature2'] in numeric_cols else (1 if len(numeric_cols) > 1 else 0), key="feature2")
            k = st.slider("Number of Clusters (k):", 2, 10, value=st.session_state['k'], key="k")
            reduction_method = st.selectbox("Dimensionality Reduction:", ("PCA", "t-SNE"), index=0, key="reduction_method")

        if st.button("Train Model", key="train_model"):
            with st.spinner("Training..."):
                if learning_type == "Supervised" and headers:
                    if not x_col or not y_col:
                        st.error("Please select input and target columns.")
                    else:
                        data = df[x_col + [y_col]].dropna()
                        if len(data) < 10:
                            st.error("Insufficient data (need at least 10 rows).")
                        else:
                            numeric_x_cols = [col for col in x_col if pd.api.types.is_numeric_dtype(data[col])]
                            categorical_x_cols = [col for col in x_col if col not in numeric_x_cols]
                            if not numeric_x_cols:
                                st.error("No numeric input features selected. Please choose at least one numeric column for X.")
                            else:
                                for col in numeric_x_cols:
                                    q1 = np.percentile(data[col], 25)
                                    q3 = np.percentile(data[col], 75)
                                    iqr = q3 - q1
                                    data = data[(data[col] >= q1 - 1.5 * iqr) & (data[col] <= q3 + 1.5 * iqr)]
                                X = data[x_col].values
                                y = data[y_col].values
                                if len(X) < 10:
                                    st.error("Insufficient data after outlier removal.")
                                else:
                                    progress_bar = st.progress(0)
                                    mdl, metric_value, cv_mean, cv_std, X_test_data, y_test, y_pred = train_model(X, y,
                                                                                                                  model_type_sup,
                                                                                                                  learning_rate,
                                                                                                                  epochs,
                                                                                                                  tuning=tuning,
                                                                                                                  feature_eng=feature_eng)

                                    progress_bar.progress(100)

                                    if mdl is None:
                                        st.error(
                                            "Model training failed due to insufficient data or invalid parameters.")


                                    elif model_type_sup.endswith("regression"):
                                        if not pd.api.types.is_numeric_dtype(df[y_col]):
                                            st.error("Target must be numeric for regression.")
                                        else:
                                            if not np.isnan(cv_mean):
                                                st.success(
                                                    f"Test MSE: {metric_value:.4f}, CV MSE: {cv_mean:.4f} ± {cv_std:.4f}")
                                            else:
                                                st.success(f"Test MSE: {metric_value:.4f}")
                                                st.warning(
                                                    "Cross-validation was skipped. This can happen if the dataset is too small for 5-fold validation.")

                                            metric_name = "Mean Squared Error (Loss)"
                                            model_name = model_type_sup.replace("regression", " Regression").title()
                                            sample_size = min(1000, len(y_test))
                                            indices = np.random.choice(len(y_test), sample_size, replace=False)
                                            fig = px.scatter(
                                                x=X_test_data[indices, 0] if len(x_col) == 1 else X_test_data[
                                                    indices].dot(range(len(x_col))), y=y_test[indices],
                                                labels={'x': x_col[0] if len(x_col) == 1 else 'Composite', 'y': y_col},
                                                title="Test vs Predicted (Sampled)")
                                            fig.add_scatter(
                                                x=X_test_data[indices, 0] if len(x_col) == 1 else X_test_data[
                                                    indices].dot(range(len(x_col))), y=y_pred[indices], mode='markers',
                                                name='Predictions')


                                    elif model_type_sup.endswith("classification"):
                                        # This is a robust check to ensure the target is suitable for binary classification
                                        if len(np.unique(y)) > 2:
                                            st.error(
                                                "Target must be binary (e.g., contain only two unique values like 0/1) for this classification model.")
                                        else:
                                            if not np.isnan(cv_mean):
                                                st.success(
                                                    f"Test Accuracy: {metric_value * 100:.2f}%, CV Accuracy: {cv_mean * 100:.2f}% ± {cv_std * 100:.2f}%")
                                            else:
                                                st.success(f"Test Accuracy: {metric_value * 100:.2f}%")
                                                st.warning(
                                                    "Cross-validation was skipped. For classification, each class must have at least 5 samples to perform stratified 5-fold cross-validation.")

                                            metric_name = "Accuracy"
                                            model_name = model_type_sup.replace("classification",
                                                                                " Classification").title()
                                            sample_size = min(1000, len(y_test))
                                            indices = np.random.choice(len(y_test), sample_size, replace=False)
                                            fig = px.scatter(
                                                x=X_test_data[indices, 0] if len(x_col) == 1 else X_test_data[
                                                    indices].dot(range(len(x_col))), y=y_test[indices],
                                                color=y_test[indices].astype(str),
                                                labels={'x': x_col[0] if len(x_col) == 1 else 'Composite', 'y': y_col},
                                                title="Test vs Predicted (Sampled)")
                                            fig.add_scatter(
                                                x=X_test_data[indices, 0] if len(x_col) == 1 else X_test_data[
                                                    indices].dot(range(len(x_col))), y=y_pred[indices], mode='markers',
                                                name='Predictions')

                                    if mdl and len(x_col) > 1 and model_type_sup.startswith("rf"):
                                        feat_imp = pd.DataFrame({'Feature': x_col, 'Importance': mdl.feature_importances_})
                                        st.subheader("Feature Importance")
                                        st.plotly_chart(px.bar(feat_imp, x='Feature', y='Importance'))

                                    if mdl:
                                        st.plotly_chart(fig, use_container_width=True)
                                        try:
                                            prompt_int = INTERPRETATION_PROMPT_TEMPLATE.format(model_type=model_name, x_column=', '.join(x_col), y_column=y_col, metric_name=metric_name, metric_value=metric_value)
                                            response_int = model.generate_content(prompt_int)
                                            cleaned_int = response_int.text.strip().replace("```json", "").replace("```", "").strip()
                                            intp = json.loads(cleaned_int)
                                            st.subheader("AI Interpretation")
                                            st.write(intp["interpretation"])
                                        except Exception as e:
                                            st.error(f"Error getting AI interpretation: {str(e)}.")

                elif headers:
                    data = df[[feature1, feature2]].dropna()
                    if len(data) < 10:
                        st.error("Insufficient data (need at least 10 rows).")
                    else:
                        scaler = StandardScaler()
                        data_scaled = scaler.fit_transform(data)
                        progress_bar = st.progress(0)
                        kmeans, labels, inertia, data_reduced = train_clustering(data_scaled, k, reduction_method)
                        progress_bar.progress(100)
                        st.success(f"Clustering complete! Inertia: {inertia:.2f} (lower is better)")
                        sample_size = min(1000, len(data_reduced))
                        sampled_indices = np.random.choice(len(data_reduced), sample_size, replace=False)
                        fig = px.scatter(x=data_reduced[sampled_indices, 0], y=data_reduced[sampled_indices, 1], color=labels[sampled_indices].astype(str), labels={'x': feature1, 'y': feature2}, title="K-Means Clustering (Sampled)")
                        centers = kmeans.cluster_centers_
                        fig.add_scatter(x=centers[:,0], y=centers[:,1], mode='markers', marker=dict(size=15, color='black'), name='Centroids')
                        st.plotly_chart(fig, use_container_width=True)

                        try:
                            prompt_int = INTERPRETATION_PROMPT_TEMPLATE.format(model_type="K-Means Clustering", x_column=feature1, y_column=feature2, metric_name="Inertia", metric_value=inertia)
                            response_int = model.generate_content(prompt_int)
                            cleaned_int = response_int.text.strip().replace("```json", "").replace("```", "").strip()
                            intp = json.loads(cleaned_int)
                            st.subheader("AI Interpretation")
                            st.write(intp["interpretation"])
                        except Exception as e:
                            st.error(f"Error getting AI interpretation: {str(e)}.")
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. CONFIGURATION ---
RANDOM_STATE = 42
TARGET_COL = 'MedHouseVal'
MODEL_FILENAME = 'best_housing_predictor.joblib'

# --- 2. TRAINING PIPELINE FUNCTION ---
def run_training_pipeline():
    """Executes the full machine learning pipeline: load, preprocess, train, and save."""
    
    st.info("Training pipeline started. This may take a moment...")
    
    # 2.1 GET DATA (California Housing)
    # The Scikit-learn California Housing dataset is used for reproducibility.
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    X = df.drop(TARGET_COL, axis=1)
    # Log-transform target for better model stability
    y = np.log1p(df[TARGET_COL]) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # 2.2 PREPROCESSING PIPELINE
    NUMERIC_FEATURES = X.columns.tolist() 
    numeric_pipeline = Pipeline(steps=[
        # Impute missing values with median
        ('imputer', SimpleImputer(strategy='median')), 
        # Scale features to mean=0, std=1
        ('scaler', StandardScaler()) 
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, NUMERIC_FEATURES)
        ],
        remainder='passthrough'
    )

    # 2.3 MODEL PIPELINE AND HYPERPARAMETER SEARCH
    # We define the full pipeline including preprocessing and model placeholder.
    model_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge()) # Initial placeholder
    ])

    # Parameter grid for GridSearchCV to find the best regularized model. [Image of machine learning pipeline components and workflow]
    param_grid = [
        {'regressor': [Ridge(random_state=RANDOM_STATE)],
         'regressor__alpha': np.logspace(-3, 3, 7)},
        {'regressor': [Lasso(random_state=RANDOM_STATE, max_iter=2000)],
         'regressor__alpha': np.logspace(-4, -2, 5)},
        {'regressor': [ElasticNet(random_state=RANDOM_STATE, max_iter=2000)],
         'regressor__alpha': [0.001, 0.01],
         'regressor__l1_ratio': [0.1, 0.5, 0.9]}
    ]

    # GridSearchCV performs cross-validated search over the parameter grid
    grid_search = GridSearchCV(
        model_pipe,
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    # 2.4 EVALUATION AND SAVING
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    # Convert back to original units for meaningful RMSE
    rmse_orig_units = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred)))

    # Save the best model
    joblib.dump(best_model, MODEL_FILENAME)
    
    st.success(f"✅ Training Complete! Best Model saved as '{MODEL_FILENAME}'.")
    st.metric("Test RMSE (in $100k units)", f"{rmse_orig_units:.4f}")
    st.write(f"Best Model Type: **{best_model['regressor'].__class__.__name__}**")
    
    # Rerun the Streamlit app to load the new model
    st.experimental_rerun()
    return best_model

# --- 3. MODEL LOADING (WITH TRAINING CHECK) ---
@st.cache_resource
def load_predictor():
    """Loads the trained model pipeline from disk."""
    try:
        model = joblib.load(MODEL_FILENAME)
        return model
    except FileNotFoundError:
        return None

# --- 4. STREAMLIT APPLICATION ---

st.set_page_config(page_title="Housing Price Dashboard", layout="wide")
st.title("🏘️ Predictive Housing Market Dashboard")

model_pipeline = load_predictor()

if model_pipeline is None:
    # If model is missing, prompt user to train
    st.error(f"⚠️ Model file '{MODEL_FILENAME}' not found. The model must be trained first.")
    st.info("Click the button below to train and save the model using the defined pipeline.")
    
    if st.button("Start Training Pipeline", type="primary"):
        # This function handles training and saves the model
        run_training_pipeline()
    st.stop() # Stop execution if model is not loaded

# --- 5. DASHBOARD UI (Different UI structure using tabs and columns) ---

st.markdown("Analyze property features and get instant predictions using the optimized machine learning model.")

# Use tabs for a cleaner, multi-section UI
tab1, tab2 = st.tabs(["🏠 Single Prediction Tool", "📊 Model Information"])

with tab1:
    st.header("Input Property Details")
    
    # Define the features and their ranges (based on California Housing)
    feature_config = {
        'MedInc': {'label': 'Median Income (10k USD)', 'min': 0.5, 'max': 15.0, 'default': 4.0},
        'HouseAge': {'label': 'House Age (Years)', 'min': 1, 'max': 52, 'default': 30},
        'AveRooms': {'label': 'Average Rooms', 'min': 1.0, 'max': 15.0, 'default': 5.5},
        'AveBedrms': {'label': 'Average Bedrooms', 'min': 0.5, 'max': 5.0, 'default': 1.1},
        'Population': {'label': 'Population', 'min': 10, 'max': 36000, 'default': 1500},
        'AveOccup': {'label': 'Average Occupancy', 'min': 1.0, 'max': 6.0, 'default': 2.5},
        'Latitude': {'label': 'Latitude', 'min': 32.5, 'max': 42.0, 'default': 34.0},
        'Longitude': {'label': 'Longitude', 'min': -124.5, 'max': -114.0, 'default': -118.2}
    }
    
    # Use columns to organize input widgets horizontally
    col_input_1, col_input_2, col_input_3 = st.columns(3)
    
    inputs = {}
    # Iterate through features to place inputs in columns
    for i, (name, config) in enumerate(feature_config.items()):
        col = [col_input_1, col_input_2, col_input_3][i % 3]
        
        with col:
            inputs[name] = st.slider(
                config['label'], 
                min_value=config['min'], 
                max_value=config['max'], 
                value=config['default'],
                step=(config['max'] - config['min']) / 100
            )

    st.markdown("---")
    
    # Compile input into a DataFrame
    FEATURE_NAMES = list(feature_config.keys())
    input_data = pd.DataFrame([inputs])[FEATURE_NAMES]
    
    if st.button("Calculate Predicted Price", key='predict_button', use_container_width=True):
        log_prediction = model_pipeline.predict(input_data)[0]
        prediction_value = np.expm1(log_prediction)
        predicted_usd = prediction_value * 100000
        
        st.success("## ✅ Prediction Complete")
        
        col_metric, col_explanation = st.columns([1, 2])
        
        with col_metric:
            st.metric(
                label="Estimated Median House Value (USD)", 
                value=f"${predicted_usd:,.0f}",
                help=f"Raw model output is {prediction_value:.2f} (in $100k units)."
            )
        
        with col_explanation:
            st.info(f"""
            This prediction was generated using the saved **{model_pipeline['regressor'].__class__.__name__}** model.
            The predicted value reflects the **median** house price for the entire block group, not a single house.
            """)

with tab2:
    st.header("Model Performance & Diagnostics")
    st.subheader("Current Loaded Model")
    
    col_metrics_2, col_metrics_3 = st.columns(2)
    with col_metrics_2:
         st.metric("Model Type", model_pipeline['regressor'].__class__.__name__)
    with col_metrics_3:
         st.metric("Training Status", "Loaded from disk") 
         
    st.subheader("Model Feature Overview")
    st.write("The model used the following features for prediction:")
    st.code(", ".join(feature_config.keys()))

    st.warning("Note: Full model metrics (RMSE/R²) and feature importance are printed to the console/log during the training phase (`run_training_pipeline()`).")

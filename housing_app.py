import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #222 !important;
        border: 2px solid #21808D; /* Teal border in light mode */
    }
    .stMetric label, .stMetric span {
        color: #222 !important;
    }

    /* Dark mode teal border override */
    @media (prefers-color-scheme: dark) {
        .stMetric {
            background-color: #222831;
            color: #f0f2f6 !important;
            border: 2px solid #32B8C6; /* Lighter teal border in dark mode */
        }
        .stMetric label, .stMetric span {
            color: #f0f2f6 !important;
        }
    }
    </style>
""", unsafe_allow_html=True)


# Load and cache the dataset
@st.cache_data
def load_data():
    """Load California Housing dataset"""
    california = fetch_california_housing()
    df = pd.DataFrame(california.data, columns=california.feature_names)
    df['MedHouseVal'] = california.target
    return df, california.feature_names

# Train models and cache them
@st.cache_resource
def train_models(X_train, X_test, y_train, y_test):
    """Train all three models and return them with metrics"""
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    metrics = {}
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    
    models['Linear Regression'] = lr
    metrics['Linear Regression'] = {
        'R2': r2_score(y_test, y_pred_lr),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        'MAE': mean_absolute_error(y_test, y_pred_lr)
    }
    
    # LASSO Regression
    lasso = Lasso(alpha=0.1, random_state=42)
    lasso.fit(X_train_scaled, y_train)
    y_pred_lasso = lasso.predict(X_test_scaled)
    
    models['LASSO'] = lasso
    metrics['LASSO'] = {
        'R2': r2_score(y_test, y_pred_lasso),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lasso)),
        'MAE': mean_absolute_error(y_test, y_pred_lasso)
    }
    
    # Ridge Regression
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge.predict(X_test_scaled)
    
    models['Ridge'] = ridge
    metrics['Ridge'] = {
        'R2': r2_score(y_test, y_pred_ridge),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
        'MAE': mean_absolute_error(y_test, y_pred_ridge)
    }
    
    return models, metrics, scaler

# Initialize session state
if 'data_loaded' not in st.session_state:
    df, feature_names = load_data()
    
    # Split the data
    X = df[feature_names]
    y = df['MedHouseVal']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train models
    models, metrics, scaler = train_models(X_train, X_test, y_train, y_test)
    
    # Store in session state
    st.session_state.df = df
    st.session_state.feature_names = feature_names
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.models = models
    st.session_state.metrics = metrics
    st.session_state.scaler = scaler
    st.session_state.data_loaded = True

# Header
st.title("🏠 California Housing Price Predictor")
st.markdown("Predict median house values using **Linear Regression**, **LASSO**, and **Ridge** models")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Predict Price", "📈 Model Comparison", "🗺️ Data Exploration", "ℹ️ About Dataset"])

# Tab 1: Predict Price
with tab1:
    st.header("Enter House Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        med_inc = st.number_input(
            "Median Income (in tens of thousands)", 
            min_value=0.0, 
            max_value=15.0, 
            value=3.5, 
            step=0.1,
            help="Median income in block group"
        )
        
        house_age = st.number_input(
            "House Age (median age in years)", 
            min_value=1, 
            max_value=52, 
            value=25, 
            step=1
        )
        
        ave_rooms = st.number_input(
            "Average Rooms (per household)", 
            min_value=1.0, 
            max_value=20.0, 
            value=5.5, 
            step=0.1
        )
        
        ave_bedrms = st.number_input(
            "Average Bedrooms (per household)", 
            min_value=0.5, 
            max_value=10.0, 
            value=1.2, 
            step=0.1
        )
    
    with col2:
        population = st.number_input(
            "Population (block group size)", 
            min_value=1, 
            max_value=10000, 
            value=1200, 
            step=50
        )
        
        ave_occup = st.number_input(
            "Average Occupancy (persons per household)", 
            min_value=0.5, 
            max_value=15.0, 
            value=3.0, 
            step=0.1
        )
        
        latitude = st.number_input(
            "Latitude", 
            min_value=32.0, 
            max_value=42.0, 
            value=37.5, 
            step=0.01
        )
        
        longitude = st.number_input(
            "Longitude", 
            min_value=-125.0, 
            max_value=-114.0, 
            value=-122.0, 
            step=0.01
        )
    
    model_choice = st.selectbox(
        "Select Model",
        options=["Linear Regression", "LASSO", "Ridge"]
    )
    
    col_btn1, col_btn2 = st.columns([1, 3])
    
    with col_btn1:
        predict_btn = st.button("🔮 Predict Price", type="primary")
    
    with col_btn2:
        random_btn = st.button("🎲 Random Example")
    
    # Random example functionality
    if random_btn:
        random_idx = np.random.randint(0, len(st.session_state.df))
        random_row = st.session_state.df.iloc[random_idx]
        
        st.session_state.random_features = {
            'MedInc': float(random_row['MedInc']),
            'HouseAge': float(random_row['HouseAge']),
            'AveRooms': float(random_row['AveRooms']),
            'AveBedrms': float(random_row['AveBedrms']),
            'Population': float(random_row['Population']),
            'AveOccup': float(random_row['AveOccup']),
            'Latitude': float(random_row['Latitude']),
            'Longitude': float(random_row['Longitude'])
        }
        st.rerun()
    
    # Predict functionality
    if predict_btn or 'random_features' in st.session_state:
        if 'random_features' in st.session_state:
            features = st.session_state.random_features
            del st.session_state.random_features
        else:
            features = {
                'MedInc': med_inc,
                'HouseAge': house_age,
                'AveRooms': ave_rooms,
                'AveBedrms': ave_bedrms,
                'Population': population,
                'AveOccup': ave_occup,
                'Latitude': latitude,
                'Longitude': longitude
            }
        
        # Create input array
        input_features = np.array([[
            features['MedInc'],
            features['HouseAge'],
            features['AveRooms'],
            features['AveBedrms'],
            features['Population'],
            features['AveOccup'],
            features['Latitude'],
            features['Longitude']
        ]])
        
        # Scale features
        input_scaled = st.session_state.scaler.transform(input_features)
        
        # Make prediction
        model = st.session_state.models[model_choice]
        prediction = model.predict(input_scaled)[0]
        
        # Display result
        st.success("✅ Prediction Complete!")
        
        result_col1, result_col2 = st.columns([2, 1])
        
        with result_col1:
            st.markdown("### Predicted Median House Value")
            st.markdown(f"## ${prediction * 100000:,.0f}")
            st.caption(f"Model used: **{model_choice}**")
        
        with result_col2:
            st.metric(
                "Prediction Confidence",
                f"{st.session_state.metrics[model_choice]['R2']:.2%}",
                f"R² Score"
            )

# Tab 2: Model Comparison
with tab2:
    st.header("Model Performance Comparison")
    
    # Metrics comparison
    st.subheader("Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    for i, (model_name, model_metrics) in enumerate(st.session_state.metrics.items()):
        with [col1, col2, col3][i]:
            st.markdown(f"**{model_name}**")
            st.metric("R² Score", f"{model_metrics['R2']:.4f}")
            st.metric("RMSE", f"{model_metrics['RMSE']:.4f}")
            st.metric("MAE", f"{model_metrics['MAE']:.4f}")
    
    # Visualizations
    st.subheader("Metrics Visualization")
    
    # Create comparison dataframe
    metrics_df = pd.DataFrame(st.session_state.metrics).T
    
    # Plot R2 scores
    fig_r2 = go.Figure(data=[
        go.Bar(
            x=metrics_df.index,
            y=metrics_df['R2'],
            marker_color=['#21808D', '#32B8C6', '#1D7480'],
            text=metrics_df['R2'].round(4),
            textposition='auto',
        )
    ])
    fig_r2.update_layout(
        title="R² Score Comparison",
        xaxis_title="Model",
        yaxis_title="R² Score",
        yaxis_range=[0, 1],
        height=400
    )
    st.plotly_chart(fig_r2, use_container_width=True)
    
    # Feature importance
    st.subheader("Feature Importance (Coefficients)")
    
    model_for_importance = st.selectbox(
        "Select model to view feature importance",
        options=["Linear Regression", "LASSO", "Ridge"],
        key="importance_model"
    )
    
    coefficients = st.session_state.models[model_for_importance].coef_
    feature_importance_df = pd.DataFrame({
        'Feature': st.session_state.feature_names,
        'Coefficient': coefficients
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    fig_importance = px.bar(
        feature_importance_df,
        x='Coefficient',
        y='Feature',
        orientation='h',
        title=f'{model_for_importance} - Feature Coefficients',
        color='Coefficient',
        color_continuous_scale='Teal'
    )
    fig_importance.update_layout(height=500)
    st.plotly_chart(fig_importance, use_container_width=True)

# Tab 3: Data Exploration
with tab3:
    st.header("Dataset Exploration")
    
    # Dataset overview
    st.subheader("Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", f"{len(st.session_state.df):,}")
    col2.metric("Features", len(st.session_state.feature_names))
    col3.metric("Median House Value", f"${st.session_state.df['MedHouseVal'].median() * 100000:,.0f}")
    
    # Display sample data
    st.subheader("Sample Data")
    st.dataframe(st.session_state.df.head(10), use_container_width=True)
    
    # Statistical summary
    st.subheader("Statistical Summary")
    st.dataframe(st.session_state.df.describe(), use_container_width=True)
    
    # Distribution plots
    st.subheader("Feature Distributions")
    
    selected_feature = st.selectbox(
        "Select feature to visualize",
        options=list(st.session_state.feature_names) + ['MedHouseVal']
    )
    
    fig_dist = px.histogram(
        st.session_state.df,
        x=selected_feature,
        nbins=50,
        title=f'Distribution of {selected_feature}',
        color_discrete_sequence=['#21808D']
    )
    fig_dist.update_layout(height=400)
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Feature Correlation Heatmap")
    
    corr_matrix = st.session_state.df.corr()
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='Teal',
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    fig_corr.update_layout(
        title="Feature Correlation Matrix",
        height=600,
        width=800
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Geographic visualization
    st.subheader("Geographic Distribution of House Prices")
    
    sample_df = st.session_state.df.sample(n=min(5000, len(st.session_state.df)))
    
    fig_map = px.scatter_mapbox(
        sample_df,
        lat='Latitude',
        lon='Longitude',
        color='MedHouseVal',
        size='MedHouseVal',
        hover_data=['MedInc', 'HouseAge', 'AveRooms'],
        color_continuous_scale='Teal',
        zoom=5,
        height=600,
        title="California Housing Prices by Location"
    )
    fig_map.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_map, use_container_width=True)

# Tab 4: About Dataset
with tab4:
    st.header("About the California Housing Dataset")
    
    st.markdown("""
    The **California Housing dataset** contains information from the 1990 California census. 
    It includes **20,640 samples** with **8 features** each, representing block groups across California.
    
    ### Features Description
    
    | Feature | Description |
    |---------|-------------|
    | **MedInc** | Median income in block group (in tens of thousands of dollars) |
    | **HouseAge** | Median house age in block group (in years) |
    | **AveRooms** | Average number of rooms per household |
    | **AveBedrms** | Average number of bedrooms per household |
    | **Population** | Block group population |
    | **AveOccup** | Average number of household members |
    | **Latitude** | Block group latitude |
    | **Longitude** | Block group longitude |
    
    **Target Variable:** `MedHouseVal` - Median house value for California districts (in hundreds of thousands of dollars)
    
    ### Models Used
    
    #### 1. Linear Regression
    Standard ordinary least squares regression with no regularization. It finds the best-fitting line 
    by minimizing the sum of squared residuals.
    
    **Pros:** Simple, interpretable, fast
    **Cons:** Can overfit with many features, sensitive to outliers
    
    #### 2. LASSO (L1 Regularization)
    Adds L1 penalty to the loss function, which forces some coefficients to exactly zero. 
    This performs automatic feature selection.
    
    **Formula:** Loss = MSE + α × Σ|coefficients|
    
    **Pros:** Feature selection, prevents overfitting, sparse models
    **Cons:** May eliminate useful features, requires tuning α
    
    #### 3. Ridge (L2 Regularization)
    Adds L2 penalty to the loss function, which shrinks coefficients toward zero but never 
    eliminates them completely.
    
    **Formula:** Loss = MSE + α × Σ(coefficients²)
    
    **Pros:** Prevents overfitting, stable with correlated features
    **Cons:** Doesn't perform feature selection, requires tuning α
    
    ### Dataset Source
    This dataset is available through scikit-learn's `fetch_california_housing()` function 
    and is commonly used for regression tasks and educational purposes.
    """)
    
    st.info("💡 **Tip:** Use the 'Predict Price' tab to experiment with different feature values and see how each model performs!")

# Footer
st.markdown("---")
st.markdown("Built by okorie")

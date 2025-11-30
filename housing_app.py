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

# --- Configuration & Setup ---

# Page configuration
st.set_page_config(
    page_title="Revised California Housing Price Predictor",
    page_icon="🏡",
    layout="wide" # Use wide layout for better space utilization
)

# Custom CSS for a modern look
st.markdown("""
    <style>
    /* Main Streamlit container adjustments */
    .stApp {
        background-color: #f7f9fc; /* Light background for contrast */
    }
    /* Header styling */
    h1 {
        color: #004d40; /* Dark Teal for headers */
    }
    /* Card-like styling for input/output */
    .stContainer {
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        background-color: white;
    }
    /* Metric styling */
    .stMetric {
        background-color: #e0f2f1; /* Very light teal background */
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #004d40; /* Stronger teal accent */
    }
    /* Primary button styling */
    .stButton>button {
        border-radius: 12px;
        border: 1px solid #004d40;
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

# Initialize session state for data, models, and training
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
    st.session_state.X_test = X_test
    st.session_state.models = models
    st.session_state.metrics = metrics
    st.session_state.scaler = scaler
    st.session_state.data_loaded = True

# --- Main App Structure (Revised UI) ---

st.title("🏡 California Housing Price Predictor")
st.markdown("A Machine Learning dashboard to predict median house values.")

# Use a two-column layout for the main prediction interface
col_input, col_output = st.columns([1, 1.5]) 

# --- Prediction Input (Left Column) ---

with col_input:
    st.header("⚙️ Enter Property Features")
    
    # Model Selector - placed above features for better visibility
    model_choice = st.selectbox(
        "**Select Regression Model**",
        options=["Linear Regression", "LASSO", "Ridge"],
        help="Choose the model to make the prediction."
    )
    
    st.subheader("Location & Income")
    # Grouping related features
    c1, c2 = st.columns(2)
    med_inc = c1.number_input(
        "Median Income (x$10k)", 
        min_value=0.0, max_value=15.0, value=3.5, step=0.1, key='med_inc_in'
    )
    latitude = c1.number_input(
        "Latitude", 
        min_value=32.0, max_value=42.0, value=37.5, step=0.01, key='lat_in'
    )
    longitude = c2.number_input(
        "Longitude", 
        min_value=-125.0, max_value=-114.0, value=-122.0, step=0.01, key='lon_in'
    )
    house_age = c2.number_input(
        "House Age (years)", 
        min_value=1, max_value=52, value=25, step=1, key='age_in'
    )
    
    st.subheader("Household Metrics")
    c3, c4 = st.columns(2)
    ave_rooms = c3.number_input(
        "Average Rooms", 
        min_value=1.0, max_value=20.0, value=5.5, step=0.1, key='rooms_in'
    )
    ave_bedrms = c3.number_input(
        "Average Bedrooms", 
        min_value=0.5, max_value=10.0, value=1.2, step=0.1, key='bedrms_in'
    )
    population = c4.number_input(
        "Population (block group)", 
        min_value=1, max_value=10000, value=1200, step=50, key='pop_in'
    )
    ave_occup = c4.number_input(
        "Average Occupancy", 
        min_value=0.5, max_value=15.0, value=3.0, step=0.1, key='occup_in'
    )

    st.markdown("---")
    
    # Prediction/Random Buttons
    col_btn1, col_btn2 = st.columns([1, 1])
    predict_btn = col_btn1.button("🔮 Predict Price", type="primary", use_container_width=True)
    random_btn = col_btn2.button("🎲 Random Example", use_container_width=True)

# --- Prediction Logic and Output (Right Column) ---

with col_output:
    st.header("Results and Model Insights")
    
    # Function to handle prediction
    def make_prediction(features):
        input_features = np.array([[
            features['MedInc'], features['HouseAge'], features['AveRooms'], features['AveBedrms'],
            features['Population'], features['AveOccup'], features['Latitude'], features['Longitude']
        ]])
        input_scaled = st.session_state.scaler.transform(input_features)
        model = st.session_state.models[model_choice]
        prediction = model.predict(input_scaled)[0]
        return prediction

    # Handle Random Example trigger
    if random_btn:
        random_idx = np.random.randint(0, len(st.session_state.df))
        random_row = st.session_state.df.iloc[random_idx]
        
        # Update inputs directly via keys
        st.session_state.med_inc_in = float(random_row['MedInc'])
        st.session_state.age_in = float(random_row['HouseAge'])
        st.session_state.rooms_in = float(random_row['AveRooms'])
        st.session_state.bedrms_in = float(random_row['AveBedrms'])
        st.session_state.pop_in = float(random_row['Population'])
        st.session_state.occup_in = float(random_row['AveOccup'])
        st.session_state.lat_in = float(random_row['Latitude'])
        st.session_state.lon_in = float(random_row['Longitude'])
        st.rerun()

    # Collect current features
    current_features = {
        'MedInc': med_inc, 'HouseAge': house_age, 'AveRooms': ave_rooms, 'AveBedrms': ave_bedrms,
        'Population': population, 'AveOccup': ave_occup, 'Latitude': latitude, 'Longitude': longitude
    }

    # Handle Prediction trigger
    if predict_btn or random_btn:
        prediction = make_prediction(current_features)
        
        # Display Predicted Value prominently
        st.subheader("💰 Predicted Median House Value")
        st.markdown(f"<h2 style='color:#004d40;'>${prediction * 100000:,.0f}</h2>", unsafe_allow_html=True)
        st.caption(f"Prediction made using the **{model_choice}** model.")
        
        st.markdown("---")
        
        # Model Performance Snapshot (for the selected model)
        selected_metrics = st.session_state.metrics[model_choice]
        st.subheader(f"Performance Metrics for {model_choice}")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("R² Score (Confidence)", f"{selected_metrics['R2']:.4f}")
        m_col2.metric("RMSE (Error)", f"{selected_metrics['RMSE']:.4f}")
        m_col3.metric("MAE (Error)", f"{selected_metrics['MAE']:.4f}")

    else:
        # Default message before prediction
        st.info("👈 Enter the property features on the left and click 'Predict Price' to see the result.")


# --- Model Comparison & Data Exploration (via Tabs below the main panel) ---

st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📊 Compare Models", "🗺️ Data Explorer", "ℹ️ Documentation"])

# Tab 1: Compare Models (Consolidated Performance and Feature Importance)
with tab1:
    st.header("Model Performance & Feature Importance")
    
    # Metrics Comparison Bar Chart (from original code)
    st.subheader("R² Score Comparison")
    metrics_df = pd.DataFrame(st.session_state.metrics).T
    fig_r2 = go.Figure(data=[
        go.Bar(
            x=metrics_df.index,
            y=metrics_df['R2'],
            marker_color=['#004d40', '#4db6ac', '#26a69a'], # Different shades of teal
            text=metrics_df['R2'].round(4),
            textposition='auto',
        )
    ])
    fig_r2.update_layout(
        title="R² Score Comparison (Higher is Better)",
        xaxis_title="Model",
        yaxis_title="R² Score",
        yaxis_range=[0, 1],
        height=400
    )
    st.plotly_chart(fig_r2, use_container_width=True)
    
    st.subheader("Feature Importance (Coefficients)")
    
    # Feature importance chart for all models side-by-side (new approach)
    
    imp_col1, imp_col2, imp_col3 = st.columns(3)
    
    for i, model_name in enumerate(["Linear Regression", "LASSO", "Ridge"]):
        with [imp_col1, imp_col2, imp_col3][i]:
            coefficients = st.session_state.models[model_name].coef_
            feature_importance_df = pd.DataFrame({
                'Feature': st.session_state.feature_names,
                'Coefficient': coefficients
            }).sort_values('Coefficient', key=abs, ascending=False)
            
            fig_importance = px.bar(
                feature_importance_df.head(5), # Only show top 5 for cleaner view
                x='Coefficient',
                y='Feature',
                orientation='h',
                title=f'**{model_name}** Top 5 Features',
                color='Coefficient',
                color_continuous_scale='Teal'
            )
            fig_importance.update_layout(height=350, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_importance, use_container_width=True)


# Tab 2: Data Exploration (Geographic Map & Distribution Focus)
with tab2:
    st.header("Geospatial & Feature Distribution Analysis")
    
    # Geographic visualization (Map first - most impactful)
    st.subheader("Geographic Distribution of Median House Values")
    
    sample_df = st.session_state.df.sample(n=min(5000, len(st.session_state.df)))
    
    fig_map = px.scatter_mapbox(
        sample_df,
        lat='Latitude',
        lon='Longitude',
        color='MedHouseVal',
        size='MedHouseVal',
        hover_data=['MedInc', 'HouseAge', 'AveRooms'],
        color_continuous_scale='Teal',
        zoom=4.5,
        height=550,
        title="California Housing Prices by Location (Sampled Data)"
    )
    fig_map.update_layout(mapbox_style="carto-positron")
    st.plotly_chart(fig_map, use_container_width=True)
    
    # Feature Distributions
    st.subheader("Feature Distributions")
    
    dist_col1, dist_col2 = st.columns([1, 2])
    
    selected_feature = dist_col1.selectbox(
        "Select feature to visualize",
        options=list(st.session_state.feature_names) + ['MedHouseVal']
    )
    
    with dist_col2:
        fig_dist = px.histogram(
            st.session_state.df,
            x=selected_feature,
            nbins=50,
            title=f'Distribution of **{selected_feature}**',
            color_discrete_sequence=['#004d40']
        )
        fig_dist.update_layout(height=400)
        st.plotly_chart(fig_dist, use_container_width=True)


# Tab 3: Documentation (Condensed Info)
with tab3:
    st.header("Dataset and Model Documentation")
    
    # Display statistical summary quickly
    st.subheader("Statistical Summary")
    st.dataframe(st.session_state.df.describe().T, use_container_width=True)
    
    st.subheader("Model Types")
    st.markdown("""
    This app uses three common regression models, all derived from **Linear Regression**:
    
    * **Linear Regression:** Standard method, minimizes Mean Squared Error (MSE).
    * **LASSO (L1 Regularization):** Adds a penalty based on the **absolute value** of coefficients. This can force some coefficients to zero, effectively performing feature selection.
        $$\\text{Loss} = \\text{MSE} + \\alpha \\times \\sum |\\text{coefficients}|$$
    * **Ridge (L2 Regularization):** Adds a penalty based on the **square** of coefficients. This shrinks coefficients toward zero without forcing them to exactly zero. Good for preventing overfitting, especially with correlated features.
        $$\\text{Loss} = \\text{MSE} + \\alpha \\times \\sum (\\text{coefficients}^2)$$
    
    **Target Variable:** `MedHouseVal` - Median house value (in hundreds of thousands of dollars).
    """)

# Footer
st.markdown("---")
st.markdown("Built by okorie.")

# California Housing Price Predictor

A complete machine learning project that predicts house prices using the California Housing dataset with Linear Regression, LASSO, and Ridge models, featuring an interactive Streamlit web application.

## 📋 Features

- **Three Regression Models**: Linear Regression, LASSO (L1), and Ridge (L2)
- **Interactive Web Interface**: Built with Streamlit
- **Real-time Predictions**: Enter house features and get instant price predictions
- **Model Comparison**: Compare performance metrics (R², RMSE, MAE)
- **Data Visualization**: 
  - Feature importance charts
  - Correlation heatmaps
  - Geographic distribution maps
  - Feature distribution plots
- **Random Examples**: Test with random examples from the dataset

## 🚀 Installation

### 1. Clone or Download the Project

Create a new folder and save `housing_app.py` in it.

### 2. Install Required Packages

```bash
pip install streamlit pandas numpy scikit-learn plotly
```

Or create a `requirements.txt` file:

```txt
streamlit==1.28.0
pandas==2.1.0
numpy==1.24.3
scikit-learn==1.3.0
plotly==5.17.0
```

Then install:

```bash
pip install -r requirements.txt
```

## 🎯 Usage

### Run the Streamlit App

```bash
streamlit run housing_app.py
```

🏃 Running Streamlit via -m
If streamlit is not recognized as a command, you can run the app using Python’s -m flag:

```bash
python -m streamlit run housing_app.py
```
or, if you have multiple versions of Python installed:

```bash
python3 -m streamlit run housing_app.py
```
This will launch the Streamlit app in your browser. Use this method if your operating system doesn’t detect the streamlit command directly.

The app will open in your default browser at `http://localhost:8501`

### Using the App

1. **Predict Price Tab**:
   - Enter house features (income, age, rooms, location, etc.)
   - Select a model (Linear Regression, LASSO, or Ridge)
   - Click "Predict Price" to get the estimated house value
   - Use "Random Example" to test with real data samples

2. **Model Comparison Tab**:
   - View R², RMSE, and MAE metrics for all three models
   - Compare model performance visually
   - See feature importance coefficients

3. **Data Exploration Tab**:
   - View dataset statistics and sample data
   - Explore feature distributions
   - Analyze correlation between features
   - Visualize geographic distribution of house prices

4. **About Dataset Tab**:
   - Learn about the California Housing dataset
   - Understand each feature
   - Read about the three regression models

## 📊 Dataset

The California Housing dataset contains 20,640 samples from the 1990 California census with 8 features:

- **MedInc**: Median income (in tens of thousands)
- **HouseAge**: Median house age
- **AveRooms**: Average rooms per household
- **AveBedrms**: Average bedrooms per household
- **Population**: Block group population
- **AveOccup**: Average household occupancy
- **Latitude**: Block group latitude
- **Longitude**: Block group longitude

**Target**: Median house value (in hundreds of thousands of dollars)

## 🤖 Models

### Linear Regression
Standard OLS regression without regularization.

### LASSO (L1 Regularization)
- Alpha (α) = 0.1
- Performs feature selection by forcing some coefficients to zero

### Ridge (L2 Regularization)
- Alpha (α) = 1.0
- Shrinks coefficients to prevent overfitting

## 📈 Performance

Typical performance metrics on the test set:
- **R² Score**: 0.58 - 0.60
- **RMSE**: 0.71 - 0.73
- **MAE**: 0.52 - 0.53

## 🛠️ Project Structure

```
california-housing-predictor/
│
├── housing_app.py          # Main Streamlit application
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 💡 Tips for Best Results

1. **Median Income** is the strongest predictor - higher income correlates with higher prices
2. **Location** (Latitude/Longitude) significantly impacts predictions
3. **House Age** has a moderate positive effect
4. Try different models to see how regularization affects predictions

## 🔧 Customization

You can modify the models by changing hyperparameters in the `train_models()` function:

```python
# Adjust LASSO alpha
lasso = Lasso(alpha=0.1, random_state=42)  # Change 0.1 to your value

# Adjust Ridge alpha
ridge = Ridge(alpha=1.0, random_state=42)  # Change 1.0 to your value
```

## 📝 Next Steps

**Extensions you can add:**
- Hyperparameter tuning with GridSearchCV
- Add more models (Random Forest, XGBoost)
- Save/load trained models with joblib
- Add prediction intervals/confidence scores
- Deploy to Streamlit Cloud for public access

## 🐛 Troubleshooting

**Import Errors**: Make sure all packages are installed:
```bash
pip install --upgrade streamlit pandas numpy scikit-learn plotly
```

**Port Already in Use**: Run on a different port:
```bash
streamlit run housing_app.py --server.port 8502
```

## 📚 Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)

## 📄 License

This project is open source and available for educational purposes.

---

**Built with Python, Streamlit, scikit-learn, and Plotly** 🎉

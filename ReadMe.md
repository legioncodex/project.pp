# 🏘️ Consolidated Housing Price Prediction Dashboard

This project is a complete, self-contained machine learning application for predicting median house values in California, built using Streamlit. It follows a robust data science pipeline, including advanced model selection and interactive deployment, all combined into a single Python script (`full_housing_app.py`).

## ✨ Features

* **Integrated Pipeline:** The application handles the full ML lifecycle: data fetching, preprocessing, training, evaluation, and model saving.
* **Model Selection:** Uses **Grid Search** to find the best performing **Regularized Linear Model** (Ridge, Lasso, or ElasticNet).
* **Persistent Model:** The best model is saved to disk (`best_housing_predictor.joblib`) for fast re-loading.
* **Interactive UI:** A polished Streamlit interface uses **tabs and columns** for clear organization, offering single-house prediction based on user input.

## 🚀 Setup and Execution

To run this application, you must first create a dedicated folder (e.g., `python/`) and place the `full_housing_app.py` and `requirements.txt` files inside it.

### 1. Install Dependencies

Install all required packages using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
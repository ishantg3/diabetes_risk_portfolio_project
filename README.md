# Diabetes Risk Prediction Pipeline

This repository contains a complete end-to-end Machine Learning project to predict diabetes risk, created as a portfolio project.

## Project Files
- `backend_training.ipynb`: Jupyter notebook containing data cleaning, exploration, and model training (Random Forest).
- `app.py`: Streamlit frontend application for interactive medical risk prediction.
- `diabetes.csv`: The dataset used for training.
- `diabetes_model.pkl`: The serialized Random Forest model.
- `scaler.pkl`: The saved StandardScaler fitted on the training data.
- `correlation_matrix.png`: Exploratory visualization of feature correlations.

## Business Objective
To develop a predictive machine learning model that accurately identifies diabetes risk based on diagnostic measurements, minimizing false negatives (improving recall) to ensure at-risk patients are not missed.

## How to Run the Web App locally
1. Ensure you have Python installed.
2. Install the required packages:
   ```bash
   pip install pandas scikit-learn streamlit numpy joblib
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

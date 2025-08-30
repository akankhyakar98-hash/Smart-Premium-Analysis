# Smart-Premium-Analysis 
Insurance companies use various factors such as age, income, health status, and claim history to estimate premiums for customers. The goal of this project is to build a machine learning model that accurately predicts insurance premiums based on customer characteristics and policy details.
#  SmartPremium: Prediction

SmartPremium is a machine learning-powered web application that predicts insurance premium costs based on customer profiles. Built with Python, scikit-learn, and Streamlit, this tool helps insurers and customers estimate realistic premium values using historical data and predictive modeling.

---

##  Problem Statement

Insurance companies often struggle to price premiums accurately due to the complexity of customer profiles and risk factors. Manual underwriting is time-consuming and prone to inconsistencies. SmartPremium aims to automate and standardize premium prediction using machine learning, enabling:

- Faster and more consistent premium estimation
- Improved transparency for customers
- Data-driven decision-making for insurers

---

## 🔄 Workflow Overview

1. **Data Collection & Cleaning**  
   - 1.2 million records of customer and policy data  
   - Missing values handled, irrelevant features dropped

2. **Feature Engineering**  
   - Manual one-hot encoding of categorical variables  
   - Date decomposition (year, month, weekday)  
   - Scaling of numeric features using `StandardScaler`

3. **Model Selection & Evaluation**  
   - Trained and compared four ML models:
     - Linear Regression  
     - Decision Tree Regressor  
     - Random Forest Regressor  
     - Gradient Boosting Regressor  
   - Evaluated using R², MAE, RMSE  
   - **Random Forest Regressor** selected for deployment due to best performance

4. **Model Saving & Deployment**  
   - Model, scaler, and column metadata saved using `joblib`  
   - Streamlit app built for real-time prediction  
   - Manual encoding logic replicated in app for consistency

---

## 🚀 Features

- Predicts insurance premiums based on customer inputs  
- Interactive Streamlit UI with sliders and dropdowns  
- Real-time predictions with realistic output values  
- Clean, modular codebase ready for deployment

---

## 🧠 Technologies Used

- Python 3.8+  
- scikit-learn  
- pandas, numpy  
- Streamlit  
- joblib

---

## 📦 Project Structure
SMART PREMIUM ANALYSIS/ │ ├── data.ipynb                  # Notebook for training and preprocessing ├── app.py                      # Streamlit app for prediction ├── smartpremium_model.pkl      # Final trained model (Random Forest) ├── smartpremium_scaler.pkl     # Scaler for numeric features ├── scaled_columns.pkl          # List of scaled columns ├── model_input_columns.pkl     # Final input columns after encoding ├── README.md                  
# Project documentation

---



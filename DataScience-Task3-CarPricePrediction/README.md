# Used Car Price Prediction

A machine learning application that predicts the estimated price of a used car based on its characteristics.

## Project Overview

This project uses an XGBoost regression model to predict used car prices.

The dataset was cleaned by removing duplicate records and normalizing categorical values. Feature engineering was performed to create useful features such as car age, brand, and engine size.

The trained model is integrated into a Streamlit application where users can enter car details and receive an estimated price.

## Features

- Data cleaning and duplicate removal
- Exploratory Data Analysis
- Feature engineering
- Multiple regression models
- XGBoost model
- Hyperparameter tuning
- Model evaluation
- Feature importance analysis
- Residual analysis
- Streamlit web application
- Input validation

## Model Performance

The final XGBoost model achieved:

- MAE: £947.16
- RMSE: £1,460.64
- R² Score: 0.8783

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- Streamlit
- Matplotlib
- Seaborn

## Input Features

The application uses information such as:

- Car title
- Registration year
- Mileage
- Previous owners
- Fuel type
- Body type
- Engine size
- Gearbox
- Emission class
- Service history
- Number of doors
- Number of seats

## Prediction

The application performs feature engineering on the user input and passes the processed data through the trained XGBoost pipeline to generate an estimated used car price.

## Disclaimer

The predicted price is an ML-based estimate and may differ from the actual market price.
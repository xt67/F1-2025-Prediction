F1-2025-Prediction
 A project which uses Machine Learning, Fastf1 API, and historical F1 race resul;ts to predict the outcome for the F1 2025 season.

 Project Overview
 This repository contains a machine learning model which predicts F1 2025 results.
 by using: 
 FastF1 API 
 2024 Racce results
 2025 qualifying session results

 Dependencies: 
 fastf1
 numpy
 pandas
 scikit-learn
 matplotlib 

 How it works:
 Data Collection: The script pulls relevant F1 data using the FastF1 API.
Preprocessing & Feature Engineering: Converts lap times, normalizes driver names, and structures race data.
Model Training: A Gradient Boosting Regressor is trained using 2024 race results.
Prediction: The model predicts race times for 2025 and ranks drivers accordingly.
Evaluation: Model performance is measured using Mean Absolute Error (MAE).

 Usage:
 run the prediction script: 
 python3 prediction1.py

 Model Performance: 
 The Mean Absolute Error (MAE) is used to evaluate how well the model predicts race times. Lower MAE values indicate more accurate predictions.

 License: 
 This project is licensed under the MIT License.

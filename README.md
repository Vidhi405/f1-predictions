# F1 Predictions

Machine learning–based predictions for Formula 1 race outcomes using FastF1 data.

This project is an early-stage experiment that uses historical race results, driver performance trends, and basic machine-learning models to predict how drivers might perform in upcoming Grands Prix — such as the Qatar GP.

# What the project does

1) Fetches real F1 race data using FastF1
2) Builds datasets across recent seasons (2021–2024)
3) Creates simple driver-form metrics (rolling points + finishing averages)
4) Trains a baseline ML model (Decision Tree)
5) Predicts whether a driver will finish in the points (P1–P10)
6) More detailed prediction features (full finishing order, faster models, etc.) are planned as development continues.

# How to run

Install these libraries: pip install fastf1 pandas numpy scikit-learn

Run the Script: python qatarModel.py

# Current Status
This is a work in progress.
The focus right now is on:
1) Improving feature engineering
2) Adding more powerful ML models (Random Forest, etc.)
3) Expanding predictions for upcoming races


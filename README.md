# Wine Quality Prediction with MLflow Tracking

This project demonstrates a complete **machine learning pipeline** for predicting wine quality using the [Wine Quality Dataset](https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv).  
It integrates **MLflow** for experiment tracking, logging, and model registry management, while maintaining a clean structure with modularized Python scripts.

---

## Features
- **Data Loading & Preprocessing**: Load dataset from a configurable source via `.env`.
- **Experiment Tracking**: Log parameters, metrics, and artifacts with MLflow.
- **Model Training**: ElasticNet regression for predicting wine quality.
- **Model Registry**: Automatically registers trained models in MLflow’s Model Registry.
- **Logging**: Detailed pipeline logging with run-specific log files for traceability.
- **Insights**: Evaluate how physicochemical properties (acidity, alcohol, etc.) affect wine quality.

---

## Example Metrics

RMSE: 0.7318

MAE: 0.5880

R²: 0.1296

---

## Insights & Results
- **Alcohol content** shows the strongest positive correlation with wine quality, meaning higher alcohol levels often lead to higher quality ratings.  
- **Volatile acidity** is negatively correlated with quality — wines with too much acidity are usually rated lower.  
- The **ElasticNet model** achieved moderate predictive power (R² ≈ 0.13), showing that while physicochemical properties explain some variance, wine quality also depends on subjective tasting factors not in the dataset.  
- Using **MLflow tracking**, we can compare different hyperparameters (`alpha`, `l1_ratio`) to see how they impact model performance over multiple runs.

---


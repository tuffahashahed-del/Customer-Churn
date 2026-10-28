
# Customer Churn Prediction

## Project Overview
This project aims to predict whether a customer will churn in the next month using machine learning classification models. It includes exploratory data analysis, data cleaning, preprocessing, model training, evaluation, and test predictions.

## Objectives
- Explore the dataset and identify data quality issues.
- Clean and preprocess the data.
- Build a baseline model and compare machine learning classifiers.
- Evaluate model performance using classification metrics.
- Generate churn probabilities and predictions for the test dataset.

## Models Used
- DummyClassifier (Baseline)
- Logistic Regression
- Random Forest Classifier

Hyperparameter tuning was performed using GridSearchCV with average precision as the scoring metric.

## Evaluation Metrics
The models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Average Precision

Because the dataset is imbalanced, accuracy alone is not sufficient to evaluate model performance.

## Data Preprocessing
- Removed duplicate records.
- Standardized inconsistent categorical values.
- Replaced unrealistic negative values in selected numerical features with missing values.
- Imputed missing values using training data.
- Encoded categorical features.
- Excluded customer identifiers and the potentially leakage-prone `case_status` column from model features.
- Split the data into training and validation sets.

## Test Predictions
The final predictions are saved in:

`outputs/test_predictions.csv`

The file contains:
- `customer_id`: Customer identifier.
- `churn_probability`: Predicted probability of churn.
- `churn_prediction`: Predicted class (0 or 1).

## Project Structure

AI-Internship-Customer-Churn/
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   └── analysis_and_training.ipynb
├── outputs/
│   ├── test_predictions.csv
│   └── visualizations/
├── main.py
├── README.md
└── requirements.txt

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## How to Run
1. Install the required dependencies.
2. Place the training and test datasets in the `data/` directory.
3. Open and run the notebook in `notebooks/`.
4. Review the evaluation results and generated predictions.

## Tools and Libraries
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

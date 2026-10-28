# Customer Churn Prediction Using Machine Learning

## Project Overview

This project develops a machine learning solution to predict whether a telecommunications customer is likely to cancel their subscription in the following month. It covers exploratory data analysis, data cleaning, preprocessing, model training, hyperparameter tuning, evaluation, and final test predictions.

The dataset is intentionally designed to contain data quality issues, making data investigation and preprocessing important parts of the project.

## Objectives

- Explore the dataset and identify data quality issues.
- Clean and preprocess the data using appropriate techniques.
- Build a baseline model and compare different classification algorithms.
- Tune model hyperparameters using validation data.
- Evaluate model performance using appropriate classification metrics.
- Generate churn probabilities and binary predictions for the test dataset.

## Dataset

The project uses the following files:

- `data/train.csv`: Labeled customer data used for model development and validation.
- `data/test.csv`: Unlabeled customer data used to generate final predictions.
- `data/submission_example.csv`: Example of the required prediction format.
- `DATA_DICTIONARY.md`: Description of the dataset features.

The target variable is `churned_next_month`:
- `0`: The customer stays.
- `1`: The customer churns.

## Exploratory Data Analysis

The training dataset initially contained 1,101 rows and 12 columns. The target distribution was imbalanced: approximately 83.92% of customers stayed, while 16.08% churned.

The exploration identified several data quality issues, including:
- Missing values in multiple features.
- 21 duplicate records.
- Negative values in numerical features where they were unrealistic.
- Inconsistent categorical values, including different spellings and letter cases.

The project includes visualizations of the target distribution and churn rates across contract types.

## Data Preprocessing

The preprocessing workflow included:

- Removing duplicate records.
- Standardizing inconsistent categorical values.
- Converting unrealistic negative values in selected numerical features to missing values.
- Imputing missing values using preprocessing steps fitted on training data.
- Encoding categorical features and scaling numerical features where required.
- Excluding `customer_id` from the model features.
- Excluding `case_status` because it may contain information related to the target and could cause data leakage.
- Splitting the labeled data into training and validation sets.

Preprocessing was fitted using training data and then applied to validation data to reduce the risk of data leakage.

## Models and Training

Three initial models were compared:

- **DummyClassifier:** Baseline model.
- **Logistic Regression:** Linear classification model.
- **Random Forest Classifier:** Tree-based ensemble model.

Hyperparameter tuning was performed using `GridSearchCV` with average precision as the scoring metric. The final test dataset was not used for model training or hyperparameter selection.

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Average precision (PR-AUC)

### Initial Model Comparison

| Model | Accuracy | Precision | Recall | F1-score | Average Precision |
|---|---:|---:|---:|---:|---:|
| DummyClassifier | 84.26% | 0.00% | 0.00% | 0.00% | — |
| Logistic Regression | 83.80% | 42.86% | 8.82% | 14.63% | 0.4472 |
| Random Forest | 86.11% | 83.33% | 14.71% | 25.00% | 0.3918 |

The baseline achieved relatively high accuracy because it predicted every validation customer as a non-churner. However, it failed to identify customers who churned. This demonstrates why accuracy alone is insufficient for an imbalanced dataset.

### Tuned Model Comparison

| Model | Accuracy | Precision | Recall | F1-score | Average Precision |
|---|---:|---:|---:|---:|---:|
| Tuned Logistic Regression (threshold 0.5) | 74.07% | 33.33% | 64.71% | 44.00% | 0.4532 |
| Tuned Random Forest (default prediction threshold) | 82.41% | 41.67% | 29.41% | 34.48% | 0.4410 |

Tuned Logistic Regression identified more churners and achieved a higher F1-score, while tuned Random Forest achieved higher accuracy and precision in this validation comparison.

### Threshold Selection

For the tuned Logistic Regression model, thresholds of 0.3, 0.4, 0.5, and 0.6 were compared using validation data.

Among these tested thresholds, **0.6 achieved the highest F1-score of 46.15%**, with:
- Precision: 40.91%
- Recall: 52.94%
- F1-score: 46.15%

The threshold was selected based on the validation F1-score, balancing precision and recall. A lower threshold could identify more potential churners but would also increase false positives.

## Final Test Predictions

The final predictions are saved in:

`outputs/test_predictions.csv`

The file contains exactly these columns in the required order:

1. `customer_id`: Customer identifier.
2. `churn_probability`: Predicted probability of churn, between 0 and 1.
3. `churn_prediction`: Predicted class, either 0 or 1.

The file contains 270 predictions, preserving the original customer order from `test.csv`. The binary predictions use a threshold of 0.6.

## Project Structure

```text
Customer-Churn/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── submission_example.csv
├── notebooks/
│   └── analysis_and_training.ipynb
├── outputs/
│   ├── test_predictions.csv
│   └── visualizations/
│       ├── churn_by_contract.png
│       └── churn_distribution.png
├── .gitignore
├── DATA_DICTIONARY.md
├── main.py
├── README.md
└── requirements.txt
```

## Installation

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

## How to Run

1. Clone or download this repository.
2. Install the required dependencies.
3. Ensure the dataset files are in the `data/` directory.
4. Open `notebooks/analysis_and_training.ipynb` in Jupyter Notebook or VS Code.
5. Run the notebook cells in order to reproduce the analysis, model training, evaluation, and predictions.

## Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Limitations

The models were evaluated on a validation split from the provided training dataset. The final test dataset has no target labels, so its predictive performance cannot be evaluated directly from this project alone. The selected threshold reflects validation results and may need further adjustment based on the business cost of false positives and false negatives.

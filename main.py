import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.dummy import DummyClassifier   
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve
from pathlib import Path
VISUALS_DIR = Path("outputs/visualizations")
VISUALS_DIR.mkdir(parents=True, exist_ok=True)
#task1
train_data=pd.read_csv(r"C:\Users\USER\OneDrive\Desktop\SDK\data\train.csv")
print("the data dimention",train_data.shape)
print("the column name:",train_data.columns)
print("the data type for each column:",train_data.dtypes)
print("first rows",train_data.head())
#======================================================================selecting the numerical and categorical features
identifier=train_data["customer_id"].name
target_feature=train_data["churned_next_month"].name
numerical_features=train_data.select_dtypes(include=['int64','float64']).columns.drop(target_feature)
categorical_features=train_data.select_dtypes(include=['object']).columns.drop(identifier)

print("the identifier feature:",identifier)
print("the numerical features:",numerical_features)
print("the categorical features:",categorical_features)
print("the target feature:",target_feature)
#======================================================================checking the target feature distribution
churned_next_month=train_data["churned_next_month"].value_counts()
print("the target feature distribution:",churned_next_month)
#======================================================================calculating the percentage of target
customer_will_churned=(len(train_data[train_data["churned_next_month"]==1])/churned_next_month.sum())*100
customer_will_not_churned=(len(train_data[train_data["churned_next_month"]==0])/churned_next_month.sum())*100
print("the percentage of customer will churned:",customer_will_churned)
print("the percentage of customer will not churned:",customer_will_not_churned)
#======================================================================visualizing the target feature distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=train_data , x="churned_next_month")
plt.title("Customer Churn Distribution")
plt.xlabel("Churned Next Month")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(VISUALS_DIR / "churn_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(data=train_data, x="contract_type", y="churned_next_month")
plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(VISUALS_DIR / "churn_by_contract.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
#========================================checking data problems
missing_values=train_data.isnull().sum()
duplicated_rows=train_data.duplicated().sum()
negative_tenure=train_data[train_data["tenure_months"]<0].shape[0]
negative_weekly_usage_gb =train_data[train_data["weekly_usage_gb"]<0].shape[0]

#categorical features checking
contract_type_values=train_data["contract_type"].unique()
auto_pay_values=train_data["auto_pay"].unique()
internet_type_values=train_data["internet_type"].unique()
region_values=train_data["region"].unique()
case_status_values=train_data["case_status"].unique()
churned_next_month_values=train_data["churned_next_month"].unique()


print("the missing values for each column:",missing_values)
print("the duplicated rows:",duplicated_rows)
print("the negative tenure months:",negative_tenure)
print("the negative weekly usage gb:",negative_weekly_usage_gb)
print("the unique values for contract_type:",contract_type_values)
print("the unique values for auto_pay:",auto_pay_values)
print("the unique values for internet_type:",internet_type_values)
print("the unique values for region:",region_values)
print("the unique values for case_status:",case_status_values)
print("the unique values for churned_next_month:",churned_next_month_values)

#=========================================data cleaning
#removing the duplicated rows
cleaned_df=train_data.copy()
cleaned_df=cleaned_df.drop_duplicates()
print("the shape of the data after removing the duplicated rows:",cleaned_df.shape)
#standardizing the categorical features
cleaned_df["region"]=cleaned_df["region"].str.strip().str.lower()
cleaned_df["internet_type"]=cleaned_df["internet_type"].str.strip().str.lower()
cleaned_df["auto_pay"]=cleaned_df["auto_pay"].str.strip().str.lower().replace({"1": "yes","y": "yes","0": "no","n": "no"})
cleaned_df["contract_type"]=cleaned_df["contract_type"].str.strip().str.lower().replace({"1 year": "one year","2 years": "two years","m2m": "month-to-month","monthly": "month-to-month"}   )
print("the unique values for contract_type after standardization:",cleaned_df["contract_type"].unique())
print("the unique values for auto_pay after standardization:",cleaned_df["auto_pay"].unique())
print("the unique values for internet_type after standardization:",cleaned_df["internet_type"].unique())
print("the unique values for region after standardization:",cleaned_df["region"].unique())
# cleaning invalid values for tenure_months and weekly_usage_gb
print("the values of tenure_months less than 0:",cleaned_df[cleaned_df["tenure_months"]<0])
print("the values of weekly_usage_gb less than 0:",cleaned_df[cleaned_df["weekly_usage_gb"]<0])
cleaned_df.loc[cleaned_df["tenure_months"]<0, "tenure_months"]=np.nan
cleaned_df.loc[cleaned_df["weekly_usage_gb"] < 0, "weekly_usage_gb"] = np.nan
#invalid values became nan values


# checking the missing values
columns_with_missing_values = cleaned_df.columns[cleaned_df.isnull().any()]
def check_missing_values(columns_with_missing_values):
    for col in columns_with_missing_values:
        missing_count = cleaned_df[col].isnull().sum()
        missing_percentage = (missing_count / len(cleaned_df)) * 100
        print(f"Column '{col}' has {missing_count} missing values ({missing_percentage:.2f}%).")
        if cleaned_df[col].dtype == 'int64' or cleaned_df[col].dtype == 'float64':
                print(f"Column '{col}' min: {cleaned_df[col].min()}, max: {cleaned_df[col].max()}")
                print(f"Column '{col}' quantiles:\n{cleaned_df[col].quantile([0.25, 0.5, 0.75])}")
        else:
                print(f"Column '{col}' counts:\n{cleaned_df[col].value_counts()}")
check_missing_values(columns_with_missing_values)

print("iqr for monthly_bill:",cleaned_df["monthly_bill"].quantile(0.75)-cleaned_df["monthly_bill"].quantile(0.25))
upper_limit = cleaned_df["monthly_bill"].quantile(0.75)+1.5*(cleaned_df["monthly_bill"].quantile(0.75)-cleaned_df["monthly_bill"].quantile(0.25))
print('upper limit for monthly_bill:', upper_limit)
#for monthly_bill, the upper limit is 113.90625, and the maximum value is 1001.13, which is greater than the upper limit, indicating that there are outliers in the monthly_bill column.
#required features for the model
cleaned_df.loc[ cleaned_df["monthly_bill"] > 113.90625,"monthly_bill"].sort_values(ascending=False)

outliers = cleaned_df[cleaned_df["monthly_bill"] > upper_limit]
print("Number of records above upper_limit:", len(outliers))
print("\nRecords above upper_limit:")
print( outliers[["customer_id", "monthly_bill"]].sort_values(by="monthly_bill", ascending=False))


#=================================================splitting the data into train and Vvalidation sets
input_features = ['tenure_months', 'weekly_usage_gb', 'contract_type', 'auto_pay', 'internet_type', 'region',"monthly_bill","support_tickets_90d","late_payments_6m"]
target_feature = 'churned_next_month'
train_data, validation_data = train_test_split(cleaned_df, test_size=0.2, random_state=42, stratify=cleaned_df[target_feature])
x_train,x_validation=train_data[input_features],validation_data[input_features]
y_train,y_validation=train_data[target_feature],validation_data[target_feature]


#===============================================preprocessing the data
def preprocess_data(x_train, x_validation):
    numerical_features = ['tenure_months', 'weekly_usage_gb','support_tickets_90d', 'late_payments_6m']
    categorical_features = ['contract_type', 'auto_pay', 'internet_type', 'region']
    numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
    bill_upper_limit = x_train['monthly_bill'].quantile(0.75) + 1.5 * (x_train['monthly_bill'].quantile(0.75) - x_train['monthly_bill'].quantile(0.25))
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore'))])

    monthly_bill_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('cap', FunctionTransformer(lambda x: np.clip(x, None, bill_upper_limit),feature_names_out='one-to-one'))])
    preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_features), ('bill', monthly_bill_transformer, ['monthly_bill']), ('cat', categorical_transformer, categorical_features)])
    x_train_processed = preprocessor.fit_transform(x_train)
    x_validation_processed = preprocessor.transform(x_validation)

    return x_train_processed, x_validation_processed, preprocessor
x_train_processed, x_validation_processed, preprocessor = preprocess_data(x_train, x_validation)
print("Training shape:", x_train_processed.shape)
print("Validation shape:", x_validation_processed.shape)
print("Same number of columns:",x_train_processed.shape[1] == x_validation_processed.shape[1])
print("Missing values in training:", np.isnan(x_train_processed).sum())
print("Missing values in validation:", np.isnan(x_validation_processed).sum())

print("Training max monthly_bill:",
      preprocessor.named_transformers_['bill'].transform(
          x_train[['monthly_bill']]
      ).max())

print("Validation max monthly_bill:",
      preprocessor.named_transformers_['bill'].transform(
          x_validation[['monthly_bill']]
      ).max())
#===============================================training
#building a baseline model
baseline_model = DummyClassifier(strategy='most_frequent', random_state=42)
baseline_model.fit(x_train_processed, y_train)
y_pred = baseline_model.predict(x_validation_processed)

accuracy = accuracy_score(y_validation, y_pred)
precision = precision_score(y_validation, y_pred, zero_division=0)
recall = recall_score(y_validation, y_pred, zero_division=0)
f1 = f1_score(y_validation, y_pred, zero_division=0)
cm = confusion_matrix(y_validation, y_pred, labels=[0, 1])

print("Baseline Model Performance:")
print("Accuracy:", accuracy)
print("Precision:",precision)
print("Recall:", recall)
print("F1 Score:",f1)
print("Confusion Matrix:")
print(cm)

#logistic regression model
logistic_model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000, random_state=42))])
logistic_model.fit(x_train, y_train)    
logistic_y_pred = logistic_model.predict(x_validation)
logistic_accuracy = accuracy_score(y_validation, logistic_y_pred)
logistic_precision = precision_score(y_validation, logistic_y_pred, zero_division=0)
logistic_recall = recall_score(y_validation, logistic_y_pred, zero_division=0)
logistic_f1 = f1_score(y_validation, logistic_y_pred, zero_division=0)
logistic_y_proba = logistic_model.predict_proba(x_validation)[:, 1]
logistic_pr_auc = average_precision_score(y_validation, logistic_y_proba)


print("\nLogistic Regression Model Performance:")
print("Accuracy:", logistic_accuracy)
print("Precision:", logistic_precision)
print("Recall:", logistic_recall)
print("F1 Score:", logistic_f1)
print("PR-AUC:", logistic_pr_auc)
print("Confusion Matrix:")
print(confusion_matrix(y_validation, logistic_y_pred, labels=[0, 1]))

#random forest model
random_forest_model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))])
random_forest_model.fit(x_train, y_train)   
random_forest_y_pred = random_forest_model.predict(x_validation)
random_forest_accuracy = accuracy_score(y_validation, random_forest_y_pred) 
random_forest_precision = precision_score(y_validation, random_forest_y_pred, zero_division=0)
random_forest_recall = recall_score(y_validation, random_forest_y_pred, zero_division=0)
random_forest_f1 = f1_score(y_validation, random_forest_y_pred, zero_division=0)
rf_y_proba = random_forest_model.predict_proba(x_validation)[:, 1]
rf_pr_auc = average_precision_score(y_validation, rf_y_proba)

print("\nRandom Forest Model Performance:") 
print("Accuracy:", random_forest_accuracy)
print("Precision:", random_forest_precision)    
print("Recall:", random_forest_recall)
print("F1 Score:", random_forest_f1)
print("PR-AUC:", rf_pr_auc)
print("Confusion Matrix:")
print(confusion_matrix(y_validation, random_forest_y_pred, labels=[0, 1]))

#Logistic Regression tuning

logistic_param_grid = {'classifier__C': [0.01, 0.1, 1, 10], 'classifier__solver': ['liblinear', 'lbfgs']}
logistic_model_tuned = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))])
logistic_grid = GridSearchCV(logistic_model_tuned, param_grid=logistic_param_grid, scoring='average_precision', cv=5, n_jobs=-1)
logistic_grid.fit(x_train, y_train)
logistic_model_tuned = logistic_grid.best_estimator_
logistic_y_pred_tuned = logistic_model_tuned.predict(x_validation)
logistic_y_proba_tuned = logistic_model_tuned.predict_proba(x_validation)[:, 1]
logistic_accuracy_tuned = accuracy_score(y_validation, logistic_y_pred_tuned)
logistic_precision_tuned = precision_score(y_validation, logistic_y_pred_tuned, zero_division=0)
logistic_recall_tuned = recall_score(y_validation, logistic_y_pred_tuned, zero_division=0)
logistic_f1_tuned = f1_score(y_validation, logistic_y_pred_tuned, zero_division=0)
logistic_pr_auc_tuned = average_precision_score(y_validation, logistic_y_proba_tuned)
print("\nLogistic Regression Tuned Model Performance:")
print("Best Parameters:", logistic_grid.best_params_)
print("Accuracy:", logistic_accuracy_tuned)
print("Precision:", logistic_precision_tuned)
print("Recall:", logistic_recall_tuned)
print("F1 Score:", logistic_f1_tuned)
print("PR-AUC:", logistic_pr_auc_tuned)
print("Confusion Matrix:")
print(confusion_matrix(y_validation, logistic_y_pred_tuned, labels=[0, 1]))
thresholds = [0.3, 0.4, 0.5, 0.6]
for threshold in thresholds:
    y_pred_threshold = (logistic_y_proba_tuned >= threshold).astype(int)
    print(f"\nThreshold: {threshold}")
    print("Precision:", precision_score(y_validation, y_pred_threshold, zero_division=0))
    print("Recall:", recall_score(y_validation, y_pred_threshold, zero_division=0))
    print("F1-score:", f1_score(y_validation, y_pred_threshold, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_validation, y_pred_threshold))


#random Forest tuning

rf_param_grid = {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [None, 10, 20], 'classifier__min_samples_split': [2, 5], 'classifier__min_samples_leaf': [1, 2]}
random_forest_model_tuned = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))])
rf_grid = GridSearchCV(random_forest_model_tuned, param_grid=rf_param_grid, scoring='average_precision', cv=5, n_jobs=-1)
rf_grid.fit(x_train, y_train)
random_forest_model_tuned = rf_grid.best_estimator_
rf_y_pred_tuned = random_forest_model_tuned.predict(x_validation)
rf_y_proba_tuned = random_forest_model_tuned.predict_proba(x_validation)[:, 1]
rf_accuracy_tuned = accuracy_score(y_validation, rf_y_pred_tuned)
rf_precision_tuned = precision_score(y_validation, rf_y_pred_tuned, zero_division=0)
rf_recall_tuned = recall_score(y_validation, rf_y_pred_tuned, zero_division=0)
rf_f1_tuned = f1_score(y_validation, rf_y_pred_tuned, zero_division=0)
rf_pr_auc_tuned = average_precision_score(y_validation, rf_y_proba_tuned)
print("\nRandom Forest Tuned Model Performance:")
print("Best Parameters:", rf_grid.best_params_)
print("Accuracy:", rf_accuracy_tuned)
print("Precision:", rf_precision_tuned)
print("Recall:", rf_recall_tuned)
print("F1 Score:", rf_f1_tuned)
print("PR-AUC:", rf_pr_auc_tuned)
print("Confusion Matrix:")
print(confusion_matrix(y_validation, rf_y_pred_tuned, labels=[0, 1]))


#================================================== creating test predictions
test_data = pd.read_csv(r"data/test.csv")
test_data["region"] = test_data["region"].str.strip().str.lower()
test_data["internet_type"] = test_data["internet_type"].str.strip().str.lower()
test_data["auto_pay"] = test_data["auto_pay"].str.strip().str.lower().replace({"1": "yes", "y": "yes", "0": "no", "n": "no"})
test_data["contract_type"] = test_data["contract_type"].str.strip().str.lower().replace({"1 year": "one year", "2 years": "two years", "m2m": "month-to-month", "monthly": "month-to-month"})
test_data.loc[test_data["tenure_months"] < 0, "tenure_months"] = np.nan
test_data.loc[test_data["weekly_usage_gb"] < 0, "weekly_usage_gb"] = np.nan
test_probabilities = logistic_model_tuned.predict_proba(test_data[input_features])[:, 1]
test_predictions = (test_probabilities >= 0.6).astype(int)
submission = pd.DataFrame({
    "customer_id": test_data["customer_id"],
    "churn_probability": np.round(test_probabilities, 2),
    "churn_prediction": test_predictions
})
submission.to_csv("outputs/test_predictions.csv", index=False)
print(submission.head())
print("Predictions file shape:", submission.shape)
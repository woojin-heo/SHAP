
import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score

# Load the saved models and imputer
logreg = joblib.load('/Users/shlokkamat/Documents/Documents - Shlok’s MacBook Pro/GitHub/NUS_Proj/SHAP/models/logistic_regression_model.pkl')
rf = joblib.load('/Users/shlokkamat/Documents/Documents - Shlok’s MacBook Pro/GitHub/NUS_Proj/SHAP/models/random_forest_model.pkl')
imputer = joblib.load('/Users/shlokkamat/Documents/Documents - Shlok’s MacBook Pro/GitHub/NUS_Proj/SHAP/models/data_imputer.pkl')

# Load unseen test data
unseen_test_data = pd.read_csv('/Users/shlokkamat/Documents/Documents - Shlok’s MacBook Pro/GitHub/NUS_Proj/SHAP/data/test.csv')

# Ensure unseen test data matches feature requirements
# Perform the same preprocessing as before (replace placeholders with your columns)
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
unseen_test_data = pd.get_dummies(unseen_test_data, columns=categorical_columns, drop_first=True)
required_columns = logreg.feature_names_in_  # Ensure correct feature order and presence
for col in required_columns:
    if col not in unseen_test_data:
        unseen_test_data[col] = 0  # Add missing columns with default values
unseen_test_data = unseen_test_data[required_columns]

# Impute missing values
X_unseen = imputer.transform(unseen_test_data)

# Make predictions using both models
y_pred_logreg = logreg.predict(X_unseen)
y_pred_rf = rf.predict(X_unseen)

# Evaluate performance (if ground truth is available)
# Replace 'true_labels' with actual column of ground truth if available
# true_labels = unseen_test_data['true_label']
# logreg_report = classification_report(true_labels, y_pred_logreg)
# rf_report = classification_report(true_labels, y_pred_rf)
# logreg_accuracy = accuracy_score(true_labels, y_pred_logreg)
# rf_accuracy = accuracy_score(true_labels, y_pred_rf)

# Example output
# print("Logistic Regression Report:")
# print(logreg_report)
# print("Random Forest Report:")
# print(rf_report)

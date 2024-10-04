# 00. Library
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# Step 1: Create an imbalanced binary classification dataset
X, Y = make_classification(n_samples=1000, n_features=10, n_informative=2, n_redundant=8, weights=[0.9, 0.1], flip_y=0, random_state=42)
np.unique(Y, return_counts=True)


# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=42)

# Step 3:Train model using LR
# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

# Train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

report = classification_report(y_test, y_pred)
print(report)


report_dict = classification_report(y_test, y_pred, output_dict=True)

file=open("report.json","w")
file.write(str(report_dict))
file.close()
# ========================================
# Imports
# ========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import xgboost as xgb

# ========================================
# Data Loading
# ========================================
# Load the dataset
data = pd.read_csv("creditcard.csv")

print("\nFirst 5 rows of the dataset:")
print(data.head())

print("\nDataset Information:")
print(data.info())

print("\nChecking for missing values:")
print(data.isnull().sum())

# ========================================
# Exploratory Data Analysis (EDA)
# ========================================
# Statistical description
print("\nStatistical Summary:")
print(data.describe())

# Class distribution
fraud = data[data['Class'] == 1]
genuine = data[data['Class'] == 0]

print("\nFraudulent transactions:", len(fraud))
print("Genuine transactions:", len(genuine))
print("Fraud Percentage: {:.4f}%".format((len(fraud) / len(data)) * 100))

# Plot Class Distribution
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='Class', data=data, palette="Set2")
plt.title("Visualization of Labels (Class Distribution)")
plt.xticks([0, 1], ['Genuine', 'Fraud'])
plt.ylabel('Count')

# Annotate bars
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:,}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.show()

# Correlation Matrix
plt.figure(figsize=(15, 10))
sns.heatmap(data.corr(), cmap="coolwarm_r")
plt.title("Correlation Matrix")
plt.show()

# Boxplot: Amount by Class
plt.figure(figsize=(6, 4))
sns.boxplot(x="Class", y="Amount", data=data)
plt.title("Amount Distribution by Class")
plt.show()

# ========================================
# Data Preprocessing
# ========================================
# Feature Scaling
scaler = StandardScaler()
data['NormalizedAmount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

# Drop 'Time' and 'Amount' columns
data.drop(['Time', 'Amount'], axis=1, inplace=True)

# Separate features and target
X = data.drop('Class', axis=1)
Y = data['Class']

# Train-Test Split (70% train, 30% test)
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

print("\nShape of training features:", train_X.shape)
print("Shape of testing features:", test_X.shape)

# ========================================
# Modeling
# ========================================

# Model 1: Decision Tree
decision_tree = DecisionTreeClassifier(random_state=42, class_weight="balanced")
decision_tree.fit(train_X, train_Y)
dt_preds = decision_tree.predict(test_X)

# Model 2: Random Forest
random_forest = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
random_forest.fit(train_X, train_Y)
rf_preds = random_forest.predict(test_X)

# Model 3: XGBoost (handling imbalance with scale_pos_weight)
scale_pos_weight = len(genuine) / len(fraud)
xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                                   scale_pos_weight=scale_pos_weight, random_state=42)
xgb_classifier.fit(train_X, train_Y)
xgb_preds = xgb_classifier.predict(test_X)

# ========================================
# Evaluation Function
# ========================================
def evaluate_model(y_true, y_pred, model_name):
    print(f"\nModel Evaluation: {model_name}")
    print("-" * 30)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# ========================================
# Model Evaluation
# ========================================
evaluate_model(test_Y, dt_preds, "Decision Tree")
evaluate_model(test_Y, rf_preds, "Random Forest")
evaluate_model(test_Y, xgb_preds, "XGBoost")

# ========================================
# Conclusion
# ========================================
print("\nConclusion:")
print("Based on the evaluation metrics (Precision, Recall, F1-Score), XGBoost and Random Forest are expected to perform better than Decision Tree, especially on imbalanced datasets like fraud detection.")


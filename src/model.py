# Customer Churn Prediction
# Best model: Gradient Boosting (86.75% accuracy)
# Dataset: Kaggle - Churn Modelling

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load Data
df = pd.read_csv('data/Churn_Modelling.csv')

# 2. Encode Categorical Variables
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# dropping first to avoid multicollinearity
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

# 3. Feature Engineering
# customers with zero balance tend to churn more
df['BalanceZero'] = (df['Balance'] == 0).astype(int)

df['AgeGroup'] = pd.cut(
    df['Age'],
    bins=[18, 25, 35, 45, 55, 65, 75, 85, 95],
    labels=['18-25', '26-35', '36-45', '46-55', '56-65', '66-75', '76-85', '86-95']
)

df['BalanceToSalaryRatio'] = df['Balance'] / df['EstimatedSalary']

# active members with more products are less likely to churn
df['ProductUsage'] = df['NumOfProducts'] * df['IsActiveMember']

df['TenureGroup'] = pd.cut(
    df['Tenure'],
    bins=[0, 2, 5, 7, 10],
    labels=['0-2', '3-5', '6-7', '8-10']
)

# gender-geography interaction features
df['Male_Germany'] = df['Gender'] * df['Geography_Germany']
df['Male_Spain'] = df['Gender'] * df['Geography_Spain']

df = pd.get_dummies(df, columns=['AgeGroup', 'TenureGroup'], drop_first=True)

# 4. Feature Selection
features = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'Geography_Germany', 'Geography_Spain',
    'BalanceZero', 'BalanceToSalaryRatio', 'ProductUsage',
    'Male_Germany', 'Male_Spain'
] + [col for col in df.columns if 'AgeGroup_' in col or 'TenureGroup_' in col]

X = df[features]
y = df['Exited']

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Train Best Model
# tried RF, LR, SVM, KNN — Gradient Boosting gave best results at 86.75%
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Evaluate
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
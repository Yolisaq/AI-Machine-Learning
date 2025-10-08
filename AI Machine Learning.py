# =============================================
# SDG 3: Health - Mental Illness Prevalence Prediction
# =============================================

# Step 0: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# =============================================
# Step 1: Load Dataset
# =============================================
data = pd.read_csv('mental-illnesses-prevalence.csv')

# Quick overview
print("First 5 rows:\n", data.head())
print("\nDataset Info:\n")
print(data.info())
print("\nDataset Description:\n")
print(data.describe())

# =============================================
# Step 2: Clean Column Names
# =============================================
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')
print("\nCleaned Columns:\n", data.columns)

# =============================================
# Step 3: Preprocess Data
# =============================================
# Correct column names
depression_col = 'depressive_disorders_(share_of_population)_-_sex:_both_-_age:_age-standardized'
anxiety_col = 'anxiety_disorders_(share_of_population)_-_sex:_both_-_age:_age-standardized'

# Create prevalence rate (sum of depressive + anxiety disorders share)
data['prevalence_rate'] = data[depression_col] + data[anxiety_col]

# Use 75th percentile for threshold to ensure both 0 and 1 in target
threshold = data['prevalence_rate'].quantile(0.75)
data['high_risk'] = np.where(data['prevalence_rate'] > threshold, 1, 0)

# Check target distribution
print("\nHigh-risk value counts:\n", data['high_risk'].value_counts())

# Drop columns not needed for ML
X = data.drop(columns=['entity','code','year','prevalence_rate','high_risk'])
y = data['high_risk']

# Fill missing values
X = X.fillna(X.mean())

# =============================================
# Step 4: Split Data
# =============================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================================
# Step 5: Train Model
# =============================================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# =============================================
# Step 6: Evaluate Model
# =============================================
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =============================================
# Step 7: Feature Importance (Fixed Graph)
# =============================================
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importances:\n", feature_importances)

plt.figure(figsize=(10,6))
sns.barplot(x=feature_importances.values, y=feature_importances.index, palette="viridis")
plt.title('Feature Importance for Mental Illness Prevalence Prediction')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# =============================================
# Step 8: Predict New Data
# =============================================
# Ensure new_data matches X.columns exactly
new_data = pd.DataFrame([0]*len(X.columns)).T
new_data.columns = X.columns

# Assign example values
new_data[depression_col] = 0.04
new_data[anxiety_col] = 0.03
new_data['schizophrenia_disorders_(share_of_population)_-_sex:_both_-_age:_age-standardized'] = 0.01
new_data['bipolar_disorders_(share_of_population)_-_sex:_both_-_age:_age-standardized'] = 0.02
new_data['eating_disorders_(share_of_population)_-_sex:_both_-_age:_age-standardized'] = 0.01

prediction = model.predict(new_data)
print("\nPredicted High Risk:", "Yes" if prediction[0]==1 else "No")

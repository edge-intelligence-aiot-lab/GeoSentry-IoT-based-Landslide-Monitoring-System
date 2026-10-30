import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from lightgbm import LGBMClassifier

df = pd.read_csv('C:\\Users\\yanna\\OneDrive\\Documents\\School Activities\\Thesis\\GeoSentry-IoT-based-Landslide-Monitoring-System\\Model\\Modified_landslide_dataset_1.csv')

X = df[[
    'Soil_Moisture',
    'Slope_Angle',
    'Rainfall'
]]
y = df['Landslide']

# Setup
model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    num_leaves=15,
    min_child_samples=5,
    random_state=42,
    verbose=-1,
    n_jobs=-1
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Accuracy Evaluation
scores_true = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
y_pred = cross_val_predict(model, X, y, cv=skf)

print("\n5-Fold Stratified Cross-Validation Accuracy (True Labels):")
print(f"Mean Accuracy: {scores_true.mean() * 100:.2f}% (± {scores_true.std() * 100:.2f}%)")
print(f"Fold Scores: {[f'{s * 100:.2f}%' for s in scores_true]}")

print("\nClassification Report (Aggregated Folds):")
print(classification_report(y, y_pred))

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Landslide (0)', 'Landslide (1)'],
            yticklabels=['No Landslide (0)', 'Landslide (1)'])
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('LightGBM Stratified 5-Fold Confusion Matrix')
plt.tight_layout()
plt.show()

# --- Feature Importances ---
model.fit(X, y)
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(6, 5))
sns.barplot(x=feature_importances.values, y=feature_importances.index, palette='viridis')
plt.title('LightGBM Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

model = joblib.dump(model, "C:\\Users\\yanna\\OneDrive\\Documents\\School Activities\\Thesis\\GeoSentry-IoT-based-Landslide-Monitoring-System\\Model\\LightGBM\\lightgbm.joblib")
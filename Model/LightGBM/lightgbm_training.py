# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix, classification_report
from lightgbm import LGBMClassifier

# Setups of PATHS
DATASET_PATH = Path("C:\\Users\\yanna\\OneDrive\\Documents\\School Activities\\Thesis\\GeoSentry-IoT-based-Landslide-Monitoring-System\\Model\\Modified_landslide_dataset_1.csv")
MODEL_PATH = Path("C:\\Users\\yanna\\OneDrive\\Documents\\School Activities\\Thesis\\GeoSentry-IoT-based-Landslide-Monitoring-System\\Model\\LightGBM\\lightgbm.joblib")

df = pd.read_csv(DATASET_PATH)

X = df[[
    'Soil_Moisture',
    'Slope_Angle',
    'Rainfall'
]]
y = df['Landslide']

# Train_test_splitting the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

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

# Accuracy Evaluation (folds)
y_pred = np.empty(len(X_train), dtype=y_train.dtype)
predicted_mask = np.zeros(len(X_train), dtype=bool)

fold_scores = []

for fold, (train_idx, val_idx) in enumerate(
    skf.split(X_train, y_train), start=1
):
    fold_model = clone(model)

    fold_model.fit(
        X_train.iloc[train_idx],
        y_train.iloc[train_idx]
    )

    predictions = fold_model.predict(
        X_train.iloc[val_idx]
    )

    actual = y_train.iloc[val_idx]

    # Store predictions in their corresponding positions
    y_pred[val_idx] = predictions
    predicted_mask[val_idx] = True

    # Store evaluation results for each fold
    fold_scores.append({
        "Accuracy": accuracy_score(actual, predictions),
        "Precision": precision_score(
            actual, predictions, zero_division=0
        ),
        "Recall": recall_score(
            actual, predictions, zero_division=0
        ),
        "F1-Score": f1_score(
            actual, predictions, zero_division=0
        )
    })

results = pd.DataFrame(fold_scores)

print("\n5-Fold Stratified Cross-Validation Results:")

for metric in results.columns:
    values = results[metric]

    print(
        f"{metric}: {values.mean() * 100:.2f}% "
        f"(+/- {values.std(ddof=0) * 100:.2f}%)"
    )

print("\nClassification Report (Aggregated Folds):")
print(classification_report(
    y_train,
    y_pred,
    labels=[0, 1],
    target_names=["No Landslide", "Landslide"],
    zero_division=0
    ))

# Confusion Matrix
cm = confusion_matrix(y_train, y_pred, labels=[0, 1])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Landslide (0)', 'Landslide (1)'],
            yticklabels=['No Landslide (0)', 'Landslide (1)'])
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('LightGBM Stratified Confusion Matrix')
plt.tight_layout()
plt.show()

# --- Feature Importances ---
# Using the training data to fit the model and get feature importances
final_model = clone(model)
final_model.fit(X_train, y_train)
feature_importances = pd.Series(final_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(6, 5))
sns.barplot(x=feature_importances.values, y=feature_importances.index, palette='viridis')
plt.title('LightGBM Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

joblib.dump(final_model, MODEL_PATH)
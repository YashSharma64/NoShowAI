import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, f1_score, classification_report

# -- Data Preparation --
df = pd.read_csv('data/noshow_cleaned (1).csv')

X = df.drop(columns=['NoShow', 'ScheduledDay', 'AppointmentDay'], errors='ignore')
y = df['NoShow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -- Scaling --
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

os.makedirs('model', exist_ok=True)
joblib.dump(scaler, 'model/scaler.pkl')

# -- Try SMOTE for oversampling --
try:
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    X_train_scaled_res, _ = smote.fit_resample(X_train_scaled, y_train)
    print("[INFO] SMOTE applied successfully.")
    use_smote = True
except ImportError:
    print("[INFO] imbalanced-learn not installed. Using class_weight='balanced' instead.")
    X_train_res, y_train_res = X_train, y_train
    X_train_scaled_res = X_train_scaled
    use_smote = False

# -- Model Definitions --
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
}

results = []
best_f1 = -1
best_model_name = None

print("\n" + "="*60)
print("       NO-SHOW PREDICTION: MULTI-MODEL COMPARISON         ")
print("="*60)

for name, model in models.items():
    if name == "Logistic Regression":
        X_tr = X_train_scaled_res if use_smote else X_train_scaled
        X_te = X_test_scaled
        y_tr = y_train_res if use_smote else y_train
    else:
        X_tr = X_train_res
        X_te = X_test
        y_tr = y_train_res if use_smote else y_train

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    acc  = accuracy_score(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, pos_label=1)

    results.append({"Model": name, "Accuracy": round(acc, 4),
                    "F1 (No-Show)": round(f1, 4), "R2 Score": round(r2, 4)})

    joblib.dump(model, f'model/{name.lower().replace(" ", "_")}.pkl')

    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name

# -- Summary Table --
results_df = pd.DataFrame(results).sort_values("F1 (No-Show)", ascending=False)
print("\n### Model Comparison Summary ###")
print(results_df.to_string(index=False))

# -- Detailed Reports --
print("\n### Detailed Classification Reports ###")
for name in models.keys():
    model = joblib.load(f'model/{name.lower().replace(" ", "_")}.pkl')
    X_te  = X_test_scaled if name == "Logistic Regression" else X_test
    print(f"\n--- {name} ---")
    print(classification_report(y_test, model.predict(X_te), target_names=["Show", "No-Show"]))

# -- Best Model --
print(f"\n{'='*60}")
print(f"  BEST MODEL: {best_model_name}  (F1 No-Show = {best_f1:.4f})")
best_model = joblib.load(f'model/{best_model_name.lower().replace(" ", "_")}.pkl')
joblib.dump(best_model, 'model/best_model.pkl')
print(f"  Saved as: model/best_model.pkl")
print("="*60)

# import pandas as pd
# import joblib

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report

# # Load CSV dataset
# df = pd.read_csv("ml_dataset/features.csv")

# X = df.drop(columns=["label"])
# y = df["label"]

# # Feature scaling (VERY IMPORTANT for KNN)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Stratified split to preserve class ratio
# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )

# # Random Forest with class balancing
# rf = RandomForestClassifier(
#     n_estimators=100,
#     random_state=42,
#     class_weight="balanced"
# )
# rf.fit(X_train, y_train)

# # KNN
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)

# # Evaluation
# print("🔹 Random Forest Performance")
# print(classification_report(y_test, rf.predict(X_test)))

# print("🔹 KNN Performance")
# print(classification_report(y_test, knn.predict(X_test)))

# # Save models
# joblib.dump(rf, "models/rf.pkl")
# joblib.dump(knn, "models/knn.pkl")
# joblib.dump(scaler, "models/scaler.pkl")

# print("✅ ML models trained and saved successfully")
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("ml_dataset/features.csv")

X = df.drop(columns=["label"])
y = df["label"]

# -------- BALANCE DATA --------
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# -------- FEATURE SCALING --------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------- SPLIT --------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# -------- RANDOM FOREST (UPGRADED) --------
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    min_samples_split=3,
    class_weight="balanced",
    random_state=42
)
rf.fit(X_train, y_train)

# -------- KNN (UPGRADED) --------
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

# -------- CROSS-VALIDATION --------
scores = cross_val_score(rf, X_train, y_train, cv=5)
print("\n[INFO] Cross-Validated RF Accuracy:", scores.mean())

# -------- EVALUATION --------
print("\n[INFO] Random Forest Performance")
print(classification_report(y_test, rf.predict(X_test)))

print("\n[INFO] KNN Performance")
print(classification_report(y_test, knn.predict(X_test)))

# -------- SAVE MODELS --------
joblib.dump(rf, "models/rf.pkl")
joblib.dump(knn, "models/knn.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\n[SUCCESS] ML models trained and saved successfully")

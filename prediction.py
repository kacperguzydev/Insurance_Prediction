import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from config import CLEANED_DATA_PATH

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "insurance_model.pkl")

# Load cleaned data
print("Loading cleaned data...")
df = pd.read_csv(CLEANED_DATA_PATH)
print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")

# Convert categorical features to numerical codes for modeling
df_model = df.copy()
categorical_cols = ['sex', 'smoker', 'region']
for col in categorical_cols:
    df_model[col] = df_model[col].astype('category').cat.codes

# Target variable: insuranceclaim (1 if yes, 0 if no)
df_model['insuranceclaim'] = df_model['insuranceclaim'].map({'no': 0, 'yes': 1})

# Features and target
X = df_model.drop("insuranceclaim", axis=1)
y = df_model["insuranceclaim"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest
print("Training Random Forest model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print("Model training complete.")

# Evaluate model
y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Save trained model
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(clf, f)

print(f"Trained model saved to {MODEL_PATH}")

import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv("heart.csv")
df.columns = df.columns.str.strip()

print("Columns Found:", df.columns)


df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)


numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
numeric_cols.remove("num")

print("Using Features:", numeric_cols)

X = df[numeric_cols]
y = df["num"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(n_estimators=300))
])

pipeline.fit(X_train, y_train)

print("\nCross Validation Accuracy:",
      cross_val_score(pipeline, X, y, cv=5).mean())

print("Test Accuracy:",
      pipeline.score(X_test, y_test))

y_pred = pipeline.predict(X_test)

print("\nClassification Report:\n",
      classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()


joblib.dump({
    "model": pipeline,
    "features": numeric_cols
}, "heart_model.pkl")

print("\n✅ Professional model saved successfully!")
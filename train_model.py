#this code is full ML backend code and converting this into pkl file for using in frontend and Fast-API(api endpoint building)

import pandas as pd
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

# Load data
df = sns.load_dataset("titanic")

# Cleaning
df.drop(
    ["deck", "embark_town", "alive", "class", "who", "adult_male"],
    axis=1,
    inplace=True
)

df["age"].fillna(df["age"].mean(), inplace=True)
df.dropna(subset=["embarked"], inplace=True)

le = LabelEncoder()
df["sex"] = le.fit_transform(df["sex"])
df["embarked"] = le.fit_transform(df["embarked"])
df = df.astype(int)

X = df.drop("survived", axis=1)
y = df["survived"]

x_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline
model_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ))
])

model_pipeline.fit(x_train, y_train)

# Save model
joblib.dump(model_pipeline, "model.pkl")

print("✅ Model trained and saved as model.pkl")


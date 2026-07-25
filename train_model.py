"""
train_model.py

Trains the AI Nudge prediction model.
"""


import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

import joblib

from pathlib import Path



# -----------------------------
# Load dataset
# -----------------------------


DATA_PATH = "data/training.csv"


data = pd.read_csv(DATA_PATH)



# -----------------------------
# Split inputs and output
# -----------------------------


X = data.drop(
    "best_nudge",
    axis=1
)


y = data["best_nudge"]



# -----------------------------
# Features
# -----------------------------


categorical_features = [

    "activity",

    "social"

]


numerical_features = [

    "sleep",

    "stress",

    "mood",

    "energy",

    "screen_time",

    "hour"

]



# -----------------------------
# Convert text into numbers
# -----------------------------


preprocessor = ColumnTransformer(

    transformers=[

        (

            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_features

        )

    ],

    remainder="passthrough"

)



# -----------------------------
# Create AI model
# -----------------------------


model = Pipeline(

    steps=[

        (

            "preprocessor",

            preprocessor

        ),

        (

            "classifier",

            RandomForestClassifier(

                n_estimators=300,

                random_state=42

            )

        )

    ]

)



# -----------------------------
# Split data
# -----------------------------


X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.25,

    random_state=42

)



# -----------------------------
# Train
# -----------------------------


model.fit(

    X_train,

    y_train

)



# -----------------------------
# Test accuracy
# -----------------------------


predictions = model.predict(

    X_test

)


accuracy = accuracy_score(

    y_test,

    predictions

)


print(
    f"Model accuracy: {accuracy:.2f}"
)



# -----------------------------
# Save model
# -----------------------------


Path("models").mkdir(
    exist_ok=True
)


joblib.dump(

    model,

    "models/nudge_model.pkl"

)


print(
    "Model saved successfully."
)
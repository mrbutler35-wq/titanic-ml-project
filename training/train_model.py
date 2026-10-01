import json
import re
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def load_and_explore_data():
    """Load and summarize the Titanic dataset."""
    print("Loading Titanic dataset...")
    df = pd.read_csv(Path(__file__).resolve().parent / "data" / "titanic.csv")

    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())

    print("\nDataset info:")
    print(df.info())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nSurvival rate by gender:")
    print(df.groupby("Sex")["Survived"].mean())

    print("\nSurvival rate by class:")
    print(df.groupby("Pclass")["Survived"].mean())

    return df


def title_from_name(name):
    """Normalize titles from the passenger name column."""
    if pd.isna(name):
        return "Other"

    match = re.search(r" ([A-Za-z]+)\.", str(name))
    if match is None:
        return "Other"

    title = match.group(1)
    title_mapping = {
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Ms": "Miss",
        "Mlle": "Miss",
        "Mme": "Mrs",
        "Master": "Master",
        "Dr": "Other",
        "Lady": "Other",
        "Countess": "Other",
        "Capt": "Other",
        "Col": "Other",
        "Don": "Other",
        "Major": "Other",
        "Rev": "Other",
        "Sir": "Other",
        "Jonkheer": "Other",
        "Dona": "Other",
    }
    return title_mapping.get(title, "Other")


def engineer_features(df):
    """Create deterministic derived features without fitting on the full dataset."""
    print("\nEngineering features...")
    data = df.copy()

    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)
    data["Title"] = data["Name"].map(title_from_name)

    data["AgeGroup"] = pd.cut(
        data["Age"],
        bins=[0, 12, 18, 35, 60, np.inf],
        labels=["Child", "Teen", "Adult", "Middle", "Senior"],
        right=True,
    )
    data["FareGroup"] = pd.cut(
        data["Fare"],
        bins=[-np.inf, 7.91, 14.45, 31.0, np.inf],
        labels=["Low", "Medium", "High", "VeryHigh"],
        right=True,
    )

    print(f"Feature-engineered data shape: {data.shape}")
    print("Derived features: FamilySize, IsAlone, Title, AgeGroup, FareGroup")
    return data


def build_model_pipeline():
    """Build a pipeline that imputes and encodes features based on the training split."""
    raw_numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    engineered_numeric_features = ["FamilySize", "IsAlone"]
    categorical_features = ["Sex", "Embarked", "Title", "AgeGroup", "FareGroup"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                raw_numeric_features + engineered_numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )
    return model


def train_model(model_data):
    """Train a Random Forest model using a train/test split."""
    print("\nTraining Random Forest model...")

    feature_columns = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "FamilySize",
        "IsAlone",
        "Title",
        "AgeGroup",
        "FareGroup",
    ]

    X = model_data[feature_columns]
    y = model_data["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    model = build_model_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nModel Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Survived", "Survived"]))

    print(f"\nConfusion matrix: \n{cm}")

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.named_steps["classifier"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(10), x="importance", y="feature")
    plt.title("Top 10 Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("models/feature_importance.png", dpi=300, bbox_inches="tight")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("models/confusion_matrix.png", dpi=300, bbox_inches="tight")

    return model, {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "test_size": int(len(y_test)),
        "confusion_matrix": cm.tolist(),
        "probability_range": [float(y_prob.min()), float(y_prob.max())],
    }, feature_columns


def save_model_artifacts(model, training_metrics, feature_names):
    """Save an end-to-end sklearn pipeline together with metadata."""
    print("\nSaving model artifacts...")

    project_root = Path(__file__).resolve().parent.parent
    training_models_dir = Path(__file__).resolve().parent / "models"
    serving_models_dir = project_root / "serving" / "models"

    training_models_dir.mkdir(parents=True, exist_ok=True)
    serving_models_dir.mkdir(parents=True, exist_ok=True)

    model_filename = training_models_dir / "titanic_model.joblib"
    joblib.dump(model, model_filename)
    print(f"Model saved to: {model_filename}")

    serving_model_filename = serving_models_dir / "titanic_model.joblib"
    joblib.dump(model, serving_model_filename)
    print(f"Serving model saved to: {serving_model_filename}")

    metadata = {
        "model_type": "RandomForestClassifier",
        "training_date": datetime.now().isoformat(),
        "accuracy": training_metrics["accuracy"],
        "precision": training_metrics["precision"],
        "recall": training_metrics["recall"],
        "f1": training_metrics["f1"],
        "test_size": training_metrics["test_size"],
        "confusion_matrix": training_metrics["confusion_matrix"],
        "feature_names": feature_names,
        "model_parameters": model.named_steps["classifier"].get_params(),
        "preprocessing_info": {
            "imputation_strategy": {
                "Age": "median",
                "Fare": "median",
                "Embarked": "most_frequent",
                "Title": "most_frequent",
                "AgeGroup": "most_frequent",
                "FareGroup": "most_frequent",
            },
            "new_features": ["FamilySize", "IsAlone", "Title", "AgeGroup", "FareGroup"],
            "encoding": "OneHotEncoder(handle_unknown='ignore')",
        },
    }

    metadata_filename = training_models_dir / "model_metadata.json"
    with metadata_filename.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved to: {metadata_filename}")

    serving_metadata_filename = serving_models_dir / "model_metadata.json"
    with serving_metadata_filename.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Serving metadata saved to: {serving_metadata_filename}")

    sample_passenger = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 25.0,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 8.05,
        "Embarked": "S",
    }

    sample_filename = training_models_dir / "sample_input.json"
    with sample_filename.open("w", encoding="utf-8") as f:
        json.dump(sample_passenger, f, indent=2)
    print(f"Sample input saved to: {sample_filename}")

    serving_sample_filename = serving_models_dir / "sample_input.json"
    with serving_sample_filename.open("w", encoding="utf-8") as f:
        json.dump(sample_passenger, f, indent=2)
    print(f"Serving sample input saved to: {serving_sample_filename}")

    print("\nAll model artifacts saved successfully!")
    return metadata


def main():
    """Run the full training workflow and create the model artifacts."""
    print("Starting Titanic Survival Prediction Model Training")
    print("=" * 60)

    df = load_and_explore_data()
    processed_data = engineer_features(df)

    feature_columns = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "FamilySize",
        "IsAlone",
        "Title",
        "AgeGroup",
        "FareGroup",
        "Survived",
    ]
    model_data = processed_data[feature_columns].copy()

    model, training_metrics, feature_names = train_model(model_data)
    metadata = save_model_artifacts(model, training_metrics, feature_names)

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Final accuracy: {metadata['accuracy']:.4f}")
    print(f"Precision: {metadata['precision']:.4f}")
    print(f"Recall: {metadata['recall']:.4f}")
    print(f"F1-score: {metadata['f1']:.4f}")
    print(f"Confusion matrix: {metadata['confusion_matrix']}")
    print("Model artifacts saved to training/models and serving/models")


if __name__ == "__main__":
    main()
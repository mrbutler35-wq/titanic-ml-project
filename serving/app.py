import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator


MODEL_PATH = Path(__file__).resolve().parent / "models" / "titanic_model.joblib"
METADATA_PATH = Path(__file__).resolve().parent / "models" / "model_metadata.json"

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="End-to-end Titanic survival prediction API built with a Random Forest model.",
    version="1.0.0",
)

try:
    model = joblib.load(MODEL_PATH)
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    print("Model and metadata loaded successfully.")
except Exception as exc:  # pragma: no cover - runtime guard for missing artifacts
    print(f"Error loading model artifacts: {exc}")
    model = None
    metadata = None


class PassengerInput(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "Pclass": 3,
            "Sex": "male",
            "Age": 25.0,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 8.05,
            "Embarked": "S",
        }
    })

    Pclass: int = Field(..., ge=1, le=3)
    Sex: str = Field(..., min_length=3, max_length=10)
    Age: float = Field(..., ge=0, le=120)
    SibSp: int = Field(..., ge=0)
    Parch: int = Field(..., ge=0)
    Fare: float = Field(..., ge=0)
    Embarked: str = Field(..., min_length=1, max_length=1)

    @field_validator("Sex", mode="before")
    @classmethod
    def normalize_sex(cls, value):
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("Embarked", mode="before")
    @classmethod
    def normalize_embarked(cls, value):
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @field_validator("Sex")
    @classmethod
    def validate_sex(cls, value):
        if value not in {"male", "female"}:
            raise ValueError("Sex must be either 'male' or 'female'.")
        return value

    @field_validator("Embarked")
    @classmethod
    def validate_embarked(cls, value):
        if value not in {"S", "C", "Q"}:
            raise ValueError("Embarked must be one of 'S', 'C', or 'Q'.")
        return value


class PredictionOutput(BaseModel):
    survived: int
    survival_probability: float
    prediction_confidence: str
    passenger_profile: Dict[str, Any]


def build_passenger_profile(passenger: PassengerInput) -> Dict[str, Any]:
    family_size = passenger.SibSp + passenger.Parch + 1
    fare_level = "VeryHigh" if passenger.Fare > 31.0 else "High" if passenger.Fare > 14.45 else "Medium" if passenger.Fare > 7.91 else "Low"
    if passenger.Age <= 12:
        age_group = "Child"
    elif passenger.Age <= 18:
        age_group = "Teen"
    elif passenger.Age <= 35:
        age_group = "Adult"
    elif passenger.Age <= 60:
        age_group = "Middle"
    else:
        age_group = "Senior"

    if passenger.Sex == "male":
        title = "Mr"
    elif passenger.Age < 18:
        title = "Miss"
    else:
        title = "Mrs"

    return {
        "class": f"Class {passenger.Pclass}",
        "gender": passenger.Sex,
        "age_group": age_group,
        "family_size": family_size,
        "traveling_alone": family_size == 1,
        "fare_level": fare_level,
        "embarkation_port": {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}.get(passenger.Embarked, "Unknown"),
        "title": title,
    }


def determine_confidence(probability: float) -> str:
    if probability >= 0.8 or probability <= 0.2:
        return "High"
    if probability >= 0.6 or probability <= 0.4:
        return "Medium"
    return "Low"


def prepare_feature_frame(passenger_data: dict) -> pd.DataFrame:
    """Match the training-time feature engineering exactly before prediction."""
    df = pd.DataFrame([passenger_data])
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    def title_for_name(sex: str, age: float) -> str:
        if sex == "male":
            return "Mr"
        if age < 18:
            return "Miss"
        return "Mrs"

    df["Title"] = df.apply(lambda row: title_for_name(row["Sex"], row["Age"]), axis=1)

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, float("inf")],
        labels=["Child", "Teen", "Adult", "Middle", "Senior"],
        right=True,
    )
    df["FareGroup"] = pd.cut(
        df["Fare"],
        bins=[-float("inf"), 7.91, 14.45, 31.0, float("inf")],
        labels=["Low", "Medium", "High", "VeryHigh"],
        right=True,
    )

    feature_order = [
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
    missing = [feature for feature in feature_order if feature not in df.columns]
    if missing:
        raise ValueError(f"Feature preparation missing required columns: {missing}")
    return df[feature_order]


@app.get("/")
async def root():
    """Basic health check for the service."""
    return {
        "message": "Titanic Survival Prediction API",
        "status": "healthy" if model is not None else "error",
        "model_loaded": model is not None,
        "accuracy": metadata.get("accuracy") if metadata else None,
    }


@app.get("/health")
async def health():
    """Alias health endpoint for service checks."""
    return await root()


@app.post("/predict", response_model=PredictionOutput)
async def predict_survival(passenger: PassengerInput):
    """Predict survival for a single passenger."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        dataframe = prepare_feature_frame(passenger.model_dump())
        probabilities = model.predict_proba(dataframe)
        prediction = model.predict(dataframe)[0]
        survival_probability = float(probabilities[0, 1])

        return PredictionOutput(
            survived=int(prediction),
            survival_probability=survival_probability,
            prediction_confidence=determine_confidence(survival_probability),
            passenger_profile=build_passenger_profile(passenger),
        )
    except Exception as exc:  # pragma: no cover - runtime safety net
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(exc)}") from exc


@app.post("/predict-batch")
async def predict_batch(passengers: List[PassengerInput]):
    """Predict survival for multiple passengers in a single request."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        feature_rows = [prepare_feature_frame(p.model_dump()) for p in passengers]
        dataframe = pd.concat(feature_rows, ignore_index=True)
        probabilities = model.predict_proba(dataframe)
        predictions = model.predict(dataframe)

        results = []
        for index, passenger in enumerate(passengers):
            probability = float(probabilities[index, 1])
            results.append(
                {
                    "passenger": passenger.model_dump(),
                    "prediction": {
                        "survived": int(predictions[index]),
                        "survival_probability": probability,
                        "prediction_confidence": determine_confidence(probability),
                        "passenger_profile": build_passenger_profile(passenger),
                    },
                }
            )

        return {"predictions": results, "count": len(results)}
    except Exception as exc:  # pragma: no cover - runtime safety net
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(exc)}") from exc


@app.get("/model-info")
async def model_info():
    """Return metadata for the currently loaded model."""
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not available")

    return {
        "model_type": metadata.get("model_type"),
        "training_date": metadata.get("training_date"),
        "accuracy": metadata.get("accuracy"),
        "precision": metadata.get("precision"),
        "recall": metadata.get("recall"),
        "f1": metadata.get("f1"),
        "test_size": metadata.get("test_size"),
        "features": metadata.get("feature_names"),
        "preprocessing": metadata.get("preprocessing_info"),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
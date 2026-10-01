from fastapi.testclient import TestClient

from serving.app import app


client = TestClient(app)

VALID_PASSENGER = {
    "Pclass": 1,
    "Sex": "female",
    "Age": 30.0,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 50.0,
    "Embarked": "S",
}


def test_root_health():
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert payload["message"] == "Titanic Survival Prediction API"
    assert payload["status"] in {"healthy", "error"}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in {"healthy", "error"}


def test_valid_single_prediction():
    response = client.post("/predict", json=VALID_PASSENGER)
    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {"survived", "survival_probability", "prediction_confidence", "passenger_profile"}
    assert payload["survived"] in {0, 1}
    assert 0.0 <= payload["survival_probability"] <= 1.0
    assert payload["prediction_confidence"] in {"High", "Medium", "Low"}


def test_invalid_passenger_input():
    response = client.post("/predict", json={
        "Pclass": 1,
        "Sex": "unknown",
        "Age": 30.0,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 50.0,
        "Embarked": "S",
    })
    assert response.status_code == 422


def test_batch_prediction():
    response = client.post(
        "/predict-batch",
        json=[VALID_PASSENGER, VALID_PASSENGER],
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert len(payload["predictions"]) == 2


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_type"] == "RandomForestClassifier"
    assert "accuracy" in payload
    assert "features" in payload


def test_required_features_match_training_expectations():
    response = client.get("/model-info")
    payload = response.json()
    assert payload["features"] == [
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

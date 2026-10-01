# Titanic Survival Prediction API

A portfolio-ready end-to-end machine learning project that trains a Random Forest model for Titanic survival prediction and serves it through a FastAPI application.

## Project Overview

This repository keeps the model training and model serving environments separate, which is a useful pattern for demonstrating ML engineering workflows.

- Training pipeline: `training/`
- Serving API: `serving/`
- Model artifact: saved as a reusable scikit-learn pipeline

## Current Validated Model Performance

The model was evaluated on a stratified 80/20 holdout split using the deterministic training pipeline.

- Test set size: 179 passengers
- Accuracy: 0.8436
- Precision: 0.8475
- Recall: 0.7246
- F1-score: 0.7812
- Confusion matrix: [[101, 9], [19, 50]]

Interpretation:
- TN = 101
- FP = 9
- FN = 19
- TP = 50

## Architecture

### Training

The training workflow in `training/train_model.py`:

- loads the Titanic dataset
- derives family and demographic features
- imputes missing values using train-only fit statistics
- encodes categorical variables with scikit-learn preprocessing
- trains a Random Forest classifier
- saves a fitted sklearn pipeline and metadata
- exports plots for feature importance and confusion matrix

### Serving

The FastAPI app in `serving/app.py` loads the trained pipeline and exposes the same prediction logic without re-implementing the training preprocessing manually.

## API Endpoints

- GET `/` - health status and model metadata summary
- GET `/health` - alias health check
- POST `/predict` - single passenger prediction
- POST `/predict-batch` - batch passenger prediction
- GET `/model-info` - training metadata and performance summary

## Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Pclass": 1,
    "Sex": "female",
    "Age": 30.0,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 50.0,
    "Embarked": "S"
  }'
```

## Local Development

### Install dependencies

```bash
python -m pip install -r training/requirements.txt
python -m pip install -r serving/requirements.txt
python -m pip install pytest
```

### Train the model

```bash
cd training
python train_model.py
```

### Run the API

```bash
cd serving
python app.py
```

Then open the docs at http://localhost:8000/docs.

## Validation

To run the automated tests:

```bash
pytest
```

The repository includes focused tests for:

- health endpoint
- valid prediction request
- invalid input validation
- probability range checks
- batch prediction behavior
- metadata endpoint

## Notes

This project is intended as an end-to-end ML engineering sample. It is not a production deployment and does not claim business-scale real-world impact.

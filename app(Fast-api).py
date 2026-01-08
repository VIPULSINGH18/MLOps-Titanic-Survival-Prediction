#without metrics tracking to an inference workflow.....

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Titanic Survival Prediction API")

# Load model
model = joblib.load("model.pkl")

# Input schema
class Passenger(BaseModel):
    pclass: int
    sex: int
    age: int
    sibsp: int
    parch: int
    fare: int
    embarked: int

@app.get("/")
def home():
    return {"message": "Titanic Survival Prediction API is running"}

@app.post("/predict")
def predict_survival(data: Passenger):
    input_data = np.array([[
        data.pclass,
        data.sex,
        data.age,
        data.sibsp,
        data.parch,
        data.fare,
        data.embarked
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction]

    return {
        "survived": int(prediction),
        "confidence": round(float(probability), 2)
    }



#Observability added and metrics tracking to an inference workflow.....

from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np
import time

from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST

app = FastAPI(title="Titanic Survival Prediction API")

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load("model.pkl")

# ===============================
# PROMETHEUS METRICS
# ===============================

# Count total requests
REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)

# Measure prediction latency
REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time taken for prediction"
)

# Count prediction outcomes
SURVIVED_COUNT = Counter(
    "survived_predictions_total",
    "Number of survived predictions"
)

NOT_SURVIVED_COUNT = Counter(
    "not_survived_predictions_total",
    "Number of not-survived predictions"
)

# ===============================
# INPUT SCHEMA
# ===============================
class Passenger(BaseModel):
    pclass: int
    sex: int
    age: int
    sibsp: int
    parch: int
    fare: int
    embarked: int


# ===============================
# ROUTES
# ===============================
@app.get("/")
def home():
    return {"message": "Titanic Survival Prediction API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict")
def predict_survival(data: Passenger):
    start_time = time.time()
    REQUEST_COUNT.inc()

    input_data = np.array([[
        data.pclass,
        data.sex,
        data.age,
        data.sibsp,
        data.parch,
        data.fare,
        data.embarked
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction]

    # Track outcome
    if prediction == 1:
        SURVIVED_COUNT.inc()
    else:
        NOT_SURVIVED_COUNT.inc()

    REQUEST_LATENCY.observe(time.time() - start_time)

    return {
        "survived": int(prediction),
        "confidence": round(float(probability), 2)
    }


# ===============================
# METRICS ENDPOINT
# ===============================
@app.get("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


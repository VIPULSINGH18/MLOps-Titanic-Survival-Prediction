# MLOps-Titanic-Survival-Prediction
End-to-end MLOps project implementing a production-ready ML pipeline with preprocessing, evaluation, and observability. Includes quality-gated deployment using F1-score comparison against a baseline model, FastAPI-based inference, Prometheus metrics, and Docker-ready architecture.


📘 README.md
Overview

In this project, I implemented an end-to-end machine learning pipeline with MLOps best practices, focusing on model quality control, deployment readiness, and observability. The core implementation includes training a Gradient Boosting model on the Titanic dataset, evaluating it using a weighted F1-score, and enforcing a mandatory logic gate that compares the new model’s performance against a production baseline before allowing deployment. This approach was chosen to prevent performance regression and to simulate real-world model governance practices commonly used in production ML systems.

Additionally, I wrapped the approved model in a FastAPI-based inference service, added Prometheus-style metrics for observability, and designed the pipeline to be extensible toward CI/CD and model registry integrations. The focus was not just on accuracy, but on building a production-aware ML workflow.

▶️ How to Run the Code
1. Install dependencies
pip install pandas seaborn scikit-learn joblib fastapi uvicorn prometheus-client

2. Train and evaluate the model (with logic gate)
python train_and_evaluate.py

This script trains a new model.

Compares its weighted F1-score against the baseline stored in baseline_registry.json.

Saves model.pkl only if the new model meets or exceeds the baseline.

3. Start the FastAPI inference server
uvicorn app:app --reload



⚠️ Assumptions & Limitations

The production baseline is simulated using a local JSON file instead of a real model registry (e.g., MLflow).

Feature encoding is static and assumes consistent input formats at inference time.

No automated retraining or CI/CD pipeline is included (can be added).

Dataset is small and used for demonstration purposes, not real-world performance claims.

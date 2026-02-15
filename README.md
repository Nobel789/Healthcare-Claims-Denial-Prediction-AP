# Healthcare-Claims-Denial-Prediction-API
This repository demonstrates an end-to-end AI solution for reducing hospital claim denials
It includes a machine learning model to detect billing errors (like mismatched diagnosis codes) and a FastAPI microservice to serve predictions in real-time.

🧠 Project Architecture
1. Model Training (/research)
Synthetic Data Generation: Created a dataset of medical claims with realistic logical errors (e.g., performing Knee Surgery for a Common Cold).

Rule Learning: Trained a Random Forest Classifier that learned business rules automatically, identifying key drivers of denial like "Missing Prior Authorization".

2. API Deployment (/deployment)
FastAPI Framework: Built a high-performance REST API in main.py.

Data Validation: Uses Pydantic models (ClaimRequest) to ensure incoming hospital data (age, CPT codes) is valid before processing.

Explainability: The API doesn't just return "Denied"—it returns logical reasons (e.g., "Diagnosis does not support procedure") to help billing specialists fix the error.

🚀 How to Run the API
Install dependencies:

Bash
pip install fastapi uvicorn pandas scikit-learn
Run the server:

Bash
uvicorn deployment.main:app --reload
Open your browser to http://127.0.0.1:8000/docs to test the API interactively.

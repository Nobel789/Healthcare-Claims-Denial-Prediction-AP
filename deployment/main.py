from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- 1. SETUP & MODEL INITIALIZATION ---
app = FastAPI(title="Healthcare Claims Denial Predictor")

# NOTE: In production, you would load a saved model like this:
# model = joblib.load("denial_model.pkl")

# For this demo, we train a dummy model instantly on startup so the code runs standalone.
def get_trained_model():
    # Training data columns must match the 'features' DataFrame in preprocess_claim
    X_train = pd.DataFrame({
        'age': [65, 30, 50, 70, 25],
        'provider_type': [1, 0, 2, 1, 0],  # 1=Ortho, 0=General, 2=Cardio
        'service_code': [1, 0, 2, 1, 0],   # 1=Knee Surg, 0=Visit, 2=Heart Surg
        'has_pre_auth': [1, 1, 1, 0, 1],   # 0 = Missing Auth (High Risk)
        'diagnosis_match': [1, 1, 1, 0, 1] # 0 = Mismatch (High Risk)
    })
    y_train = [0, 0, 0, 1, 0] # 1 = Denied, 0 = Approved
    
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)
    return clf

# Load the model into memory
model = get_trained_model()

# --- 2. DATA MODELS (Pydantic) ---
# These define the structure of the JSON you send to the API

class Patient(BaseModel):
    id: str
    age: int
    gender: str

class Provider(BaseModel):
    provider_id: str
    specialty: str  # e.g., "Orthopedic Surgery", "General", "Cardiology"

class ClinicalData(BaseModel):
    diagnosis_codes: List[str] # e.g., ["M17.11"]
    procedure_codes: List[str] # e.g., ["29881"]

class AdminData(BaseModel):
    billed_amount: float
    has_prior_authorization: bool

class ClaimRequest(BaseModel):
    claim_id: str
    patient: Patient
    provider: Provider
    clinical_data: ClinicalData
    admin_data: AdminData

# --- 3. HELPER FUNCTIONS (The "Translator") ---
def preprocess_claim(claim: ClaimRequest):
    """
    Converts the nested JSON object into a flat DataFrame 
    that the Machine Learning model can understand.
    """
    
    # 1. Map Provider Type to Number
    specialty_map = {"General": 0, "Orthopedic Surgery": 1, "Cardiology": 2}
    # Default to 0 (General) if specialty is unknown
    provider_code = specialty_map.get(claim.provider.specialty, 0) 

    # 2. Map Service Code to Number
    procedure_map = {"99213": 0, "29881": 1, "33405": 2}
    
    # Safety check: Ensure there is at least one procedure code
    if not claim.clinical_data.procedure_codes:
        cpt = "00000" # Placeholder
    else:
        cpt = claim.clinical_data.procedure_codes[0]
        
    service_code = procedure_map.get(cpt, 0)

    # 3. Feature Engineering: Diagnosis Match Logic
    # Rule: If Procedure is Knee Surgery (29881) AND Diagnosis is NOT Knee OA (M17.11), 
    # then it is a mismatch (0). Otherwise, we assume it matches (1).
    is_match = 1
    if cpt == "29881" and "M17.11" not in claim.clinical_data.diagnosis_codes:
        is_match = 0 

    # 4. Create the Vector (DataFrame)
    # The columns MUST match the X_train structure in get_trained_model()
    features = pd.DataFrame([{
        'age': claim.patient.age,
        'provider_type': provider_code,
        'service_code': service_code,
        'has_pre_auth': int(claim.admin_data.has_prior_authorization),
        'diagnosis_match': is_match
    }])

    return features

# --- 4. THE API ENDPOINT (The "Door") ---
@app.post("/predict_denial")
def predict_claim(claim: ClaimRequest):

    try:
        # Step A: Convert JSON to Model Features
        features_df = preprocess_claim(claim)

        # Step B: Predict
        prediction_class = model.predict(features_df)[0]     # 0 (Approved) or 1 (Denied)
        
        # Get probability (confidence score)
        # predict_proba returns [[prob_0, prob_1]], we want prob_1 (denial probability)
        probability = model.predict_proba(features_df)[0][1] 

        # Step C: Explainability Logic (Why was it denied?)
        reason = "Claim looks clean."
        
        # If the model thinks it will be denied (prob > 50%)
        if probability > 0.5:
            reason = "High likelihood of denial."
            
            # Check specific flags to give a human-readable reason
            if features_df['has_pre_auth'].iloc[0] == 0:
                reason += " Reason: Missing Prior Authorization."
            elif features_df['diagnosis_match'].iloc[0] == 0:
                reason += " Reason: Diagnosis code does not support this procedure."

        # Step D: Return JSON Response
        return {
            "claim_id": claim.claim_id,
            "prediction": "DENIED" if prediction_class == 1 else "APPROVED",
            "denial_probability": round(probability, 4),
            "explanation": reason
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 5. HOW TO RUN ---
# Terminal command: 
# uvicorn main:app --reload
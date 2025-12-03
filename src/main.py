from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.predict import RiskPredictor
import os

app = FastAPI(title="Crohn's Disease Risk Predictor")

# Initialize predictor
# We expect the model to be in the models directory relative to where this script is run
# or where the working directory is set.
try:
    predictor = RiskPredictor(model_path='models/crohns_model.joblib')
except FileNotFoundError:
    print("Model not found. Please ensure 'models/crohns_model.joblib' exists.")
    predictor = None

class PatientData(BaseModel):
    age: int
    sex: int
    bmi: float
    family_crohns: int
    crp: float
    fecal_calprotectin: float
    wbc: float
    smoking_status: int
    diet_score: int
    stress_level: int
    nod2_mutation: int
    atg16l1_mutation: int

@app.get("/")
def read_root():
    return {"message": "Crohn's Disease Risk Predictor API is running."}

@app.post("/predict")
def predict_risk(patient: PatientData):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    try:
        result = predictor.predict(patient.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    if predictor is None:
        return {"status": "unhealthy", "reason": "Model not loaded"}
    return {"status": "healthy"}

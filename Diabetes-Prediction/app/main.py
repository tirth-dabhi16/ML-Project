import pandas as pd
import joblib
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from pathlib import Path

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model.pkl"

model = joblib.load(MODEL_PATH)

class DiabetesData(BaseModel):
    gender : str
    age : float = Field(gt=0,le=120)
    hypertension : int = Field(ge=0,le=1)
    heart_disease : int = Field(ge=0,le=1)
    smoking_history : str
    bmi : float = Field(gt=0)
    HbA1c_level : float = Field(gt=0)
    blood_glucose_level : int = Field(gt=0)

@app.get("/")
def home():
    return {"message" : "Diabetes prediction api"}

@app.post("/predict")
def predict_diabetes(data : DiabetesData):
    try:
        input_df = pd.DataFrame([{
        "gender" : data.gender,
        "age" : data.age,
        "hypertension" : data.hypertension,
        "heart_disease" : data.heart_disease,
        "smoking_history" : data.smoking_history,
        "bmi" : data.bmi,
        "HbA1c_level" : data.HbA1c_level,
        "blood_glucose_level" : data.blood_glucose_level
        }])
        prediction = model.predict(input_df)[0]

        if prediction == 1:
            result = "Diabetes"
        else:
            result = "Not Diabetes"
        return {
            "Result" : result
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e))

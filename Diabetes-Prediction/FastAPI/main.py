import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel,Field

app = FastAPI()

# Load trained model
classifier = joblib.load("model.pkl")

# Pydantic Model
class DiabetesData(BaseModel):
    gender: str
    age: float = Field(gt=0,le=120)
    hypertension: int = Field(ge=0,le=1)
    heart_disease: int = Field(ge=0,le=1)
    smoking_history: str
    bmi: float = Field(gt=0)
    HbA1c_level: float = Field(gt=0)
    blood_glucose_level: int = Field(gt=0)

# Routes
@app.get("/")
def index():
    return {"message": "Diabetes Prediction API"}

@app.post("/predict")
def predict_diabetes(data:DiabetesData):

    input_df = pd.DataFrame([{
        "gender": data.gender,
        "age": data.age,
        "hypertension": data.hypertension,
        "heart_disease": data.heart_disease,
        "smoking_history": data.smoking_history,
        "bmi": data.bmi,
        "HbA1c_level": data.HbA1c_level,
        "blood_glucose_level": data.blood_glucose_level
    }])

    prediction = classifier.predict(input_df)

    return {
        "prediction": "Diabetes" if prediction[0] == 1 else "Not Diabetes"
    }
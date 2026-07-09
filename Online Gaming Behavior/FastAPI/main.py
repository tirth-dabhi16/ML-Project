import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

classifier = joblib.load("online_gaming_behavior_model.pkl")

class PlayerData(BaseModel):
    Age: int = Field(gt=0, le=120)
    Gender: str
    Location: str
    GameGenre: str
    PlayTimeHours: float = Field(ge=0)
    InGamePurchases: int = Field(ge=0, le=1)
    GameDifficulty: str
    SessionsPerWeek: int = Field(ge=0)
    AvgSessionDurationMinutes: int = Field(ge=0)
    PlayerLevel: int = Field(ge=0)
    AchievementsUnlocked: int = Field(ge=0)

@app.get("/")
def index():
    return {"message": "Player Engagement Prediction API"}

@app.post("/predict")
def predict_engagement_level(data: PlayerData):

    input_df = pd.DataFrame([{
        "Age": data.Age,
        "Gender": data.Gender.title(),
        "Location": data.Location,
        "GameGenre": data.GameGenre.title(),
        "PlayTimeHours": data.PlayTimeHours,
        "InGamePurchases": data.InGamePurchases,
        "GameDifficulty": data.GameDifficulty.title(),
        "SessionsPerWeek": data.SessionsPerWeek,
        "AvgSessionDurationMinutes": data.AvgSessionDurationMinutes,
        "PlayerLevel": data.PlayerLevel,
        "AchievementsUnlocked": data.AchievementsUnlocked
    }])

    input_df["InGamePurchases"] = input_df["InGamePurchases"].astype("object") # For datatype matching

    prediction = classifier.predict(input_df)

    mapping = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    return {
         "prediction": mapping[int(prediction[0])]
    }
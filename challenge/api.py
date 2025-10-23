from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .model import DelayModel

app = FastAPI(title="SCL Flight Delay API", version="1.0.0")

# -------------------------------------------------------
# ----- Load training data & prepare allowed values -----
# -------------------------------------------------------
DATA_PATH = (Path(__file__).resolve().parents[1] / "data" / "data.csv")
MODEL = DelayModel()

try:
    df_train = pd.read_csv(DATA_PATH)
    X_train, y_train = MODEL.preprocess(df_train, target_column="delay")
    MODEL.fit(X_train, y_train)

    ALLOWED_OPERAS = set(df_train["OPERA"].dropna().unique().tolist())
    ALLOWED_TIPOVUELO = {"I", "N"}
    ALLOWED_MES = set(range(1, 13))
except Exception:
    ALLOWED_OPERAS = {"Aerolineas Argentinas", "Grupo LATAM", "Sky Airline", "Copa Air", "Latin American Wings"}
    ALLOWED_TIPOVUELO = {"I", "N"}
    ALLOWED_MES = set(range(1, 13))

# -----------------------------------------
# ---------------- Schemas ----------------
# -----------------------------------------
class PredictRequest(BaseModel):
    """
    Request body for /predict.
    - flights: list of flight records (OPERA, TIPOVUELO, MES).
    """
    flights: List[Dict[str, Any]] = Field(default_factory=list)

class PredictResponse(BaseModel):
    """
    Response body for /predict.
    - predict: list of 0/1 delay labels for each input record.
    """
    predict: List[int]

# -----------------------------------------
# ---------------- Health -----------------
# -----------------------------------------
@app.get("/health", status_code=200)
async def get_health() -> dict:
    return { "status": "OK" }

# -----------------------------------------
# ---------------- Predict ----------------
# -----------------------------------------
@app.post("/predict", status_code=200, response_model=PredictResponse)
async def post_predict(payload: PredictRequest) -> PredictResponse:
    """
    Predict delay labels for incoming flight records.

    Contract (as required by tests):
    - Input key: 'flights'
    - Output key: 'predict'

    Validation:
    - OPERA must exist in the training dataset.
    - TIPOVUELO must be one of {'I','N'}.
    - MES must be an integer in [1..12].
    """
    if not payload.flights:
        return PredictResponse(predict=[])

    for rec in payload.flights:
        opera = rec.get("OPERA")
        tipovuelo = rec.get("TIPOVUELO")
        mes = rec.get("MES")

        if opera not in ALLOWED_OPERAS:
            raise HTTPException(status_code=400, detail="Unknown OPERA")
        if tipovuelo not in ALLOWED_TIPOVUELO:
            raise HTTPException(status_code=400, detail="Unknown TIPOVUELO")
        if not isinstance(mes, int) or (mes not in ALLOWED_MES):
            raise HTTPException(status_code=400, detail="Unknown MES")

    df = pd.DataFrame(payload.flights)
    X = MODEL.preprocess(df)
    preds = MODEL.predict(X)
    return PredictResponse(predict=preds)
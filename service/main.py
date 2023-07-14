import mlflow
import pandas as pd
import pytest
from data_model import TaxiRide, TaxiRidePrediction
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from predict import predict
from dotenv import load_dotenv
import os

app = FastAPI()
load_dotenv()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

SA_KEY = os.getenv("SA_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SA_KEY    

stage = 'Production'
model_uri = f'models:/lr-ride-duration/{stage}'
MODEL = mlflow.pyfunc.load_model(model_uri)

def get_taxi_predictions(taxi_rides: list[TaxiRide]) -> pd.DataFrame:
    predictions = []
    for taxi_ride in taxi_rides:
        prediction_duration = predict(MODEL, model_uri, taxi_ride)
        prediction = TaxiRidePrediction(
            **taxi_ride.dict(), predicted_duration=prediction_duration)
        predictions.append(prediction)
    df_predictions = pd.DataFrame([prediction.dict()
                                  for prediction in predictions])
    return df_predictions


@app.get("/")
def index():
    return {"message": "NYC Taxi Ride Duration Prediction"}


@app.post("/predict", response_model=TaxiRidePrediction)
def predict_duration(data: TaxiRide):
    
    prediction = predict(MODEL, MLFLOW_TRACKING_URI,data)
    return TaxiRidePrediction(**data.dict(), predicted_duration=prediction)


@app.post("/predict_batch", response_model=list[TaxiRidePrediction])
def predict_durations(data: list[TaxiRide]):
    df_predictions = get_taxi_predictions(data)
    return df_predictions.to_dict(orient='records')


@app.post("/predict_bq")
def predict_durations_to_bigquery(data: list[TaxiRide]):
    df_predictions = get_taxi_predictions(MODEL, data)
    df_predictions.to_gbq('taxi_predictions.predictions',
                          'composed-hold-390914', if_exists='append')
    return {"message": "Successfully uploaded"}

import os

import mlflow
import pandas as pd
import uvicorn
from data_model import TaxiRide, TaxiRidePrediction
from dotenv import load_dotenv
from fastapi import FastAPI
from predict import predict

app = FastAPI()
load_dotenv()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
DEPLOYED_ON_CLOUD_RUN = os.getenv('K_SERVICE')

if DEPLOYED_ON_CLOUD_RUN is None or DEPLOYED_ON_CLOUD_RUN == '':
    SA_KEY = os.getenv("SA_KEY")
    if SA_KEY is not None:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SA_KEY

stage = 'Production'
model_uri = f'models:/lr-ride-duration/{stage}'
MODEL = mlflow.pyfunc.load_model(model_uri)


def get_taxi_predictions(taxi_rides: list[TaxiRide]) -> list:
    predictions = []
    for taxi_ride in taxi_rides:
        prediction_duration = predict(MODEL, model_uri, taxi_ride)
        prediction = TaxiRidePrediction(
            **taxi_ride.dict(), predicted_duration=prediction_duration)
        predictions.append(prediction)

    return predictions


@app.get("/")
def index():
    return {"message": "NYC Taxi Ride Duration Prediction"}


@app.post("/predict", response_model=TaxiRidePrediction)
def predict_duration(data: TaxiRide):

    prediction = predict(MODEL, MLFLOW_TRACKING_URI, data)

    return TaxiRidePrediction(**data.dict(), predicted_duration=prediction)


@app.post("/predict_batch", response_model=list[TaxiRidePrediction])
def predict_durations(data: list[TaxiRide]):
    predictions = get_taxi_predictions(data)
    return predictions


@app.post("/predict_bq")
def predict_durations_to_bigquery(data: list[TaxiRide]):
    predictions = get_taxi_predictions(data)
    df_predictions = pd.DataFrame([prediction.dict()
                                  for prediction in predictions])
    df_predictions.to_gbq('taxi_predictions.predictions',
                          'composed-hold-390914', if_exists='append')

    return {"message": "Successfully uploaded"}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=9696)


import google.cloud.bigquery as gcl
import pandas as pd
import pytest
from data_model import TaxiRide
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fixtures import expected_predictions, taxi_rides, DURATION_TOLERANCE
from google.cloud.bigquery import Table
from main import app, get_taxi_predictions

client = TestClient(app)


class TestApiEndpoints:

    def test_index(self):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {
            "message": "NYC Taxi Ride Duration Prediction"}

    def test_get_taxi_predictions(self, taxi_rides, expected_predictions):
        taxi_ride_models = []
        for ride in taxi_rides:
            taxi_ride_models.append(TaxiRide(**ride))

        output = get_taxi_predictions(taxi_ride_models)

        assert output == expected_predictions

    @pytest.mark.parametrize("input, expected_output", [(
        {
            "ride_id": "Ride-1",
            "PULocationID": 1,
            "DOLocationID": 2,
            "trip_distance": 2.5
        },
        {
            "ride_id": "Ride-1",
            "PULocationID": 1,
            "DOLocationID": 2,
            "trip_distance": 2.5,
            "predicted_duration": pytest.approx(12.745262840390389, abs=DURATION_TOLERANCE)
        }
    )])
    def test_predict_duration(self, input, expected_output):
        response = client.post('/predict', json=input)

        assert response.status_code == 200

        assert response.json() == expected_output

    def test_predict_durations(self, taxi_rides, expected_predictions):
        response = client.post('/predict_batch', json=taxi_rides)

        assert response.status_code == 200

        assert response.json() == expected_predictions

    def test_predict_durations_to_bigquery(self, taxi_rides):
        response = client.post('/predict_bq', json=taxi_rides)

        assert response.status_code == 200

        assert response.json() == {"message": "Successfully uploaded"}

        table_ref = gcl.Dataset(
            'composed-hold-390914.taxi_predictions').table('predictions')

        assert type(gcl.Client().get_table(table=table_ref)) == Table

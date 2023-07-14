import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fixtures import expected_predictions, taxi_rides
from main import app, get_taxi_predictions
from pandas.testing import assert_frame_equal
from data_model import TaxiRide
import json

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

        # Test case 1 assert we get a dataframe
        assert (type(output) == pd.DataFrame)

        # Test case 2 assert
        assert_frame_equal(output, pd.DataFrame(expected_predictions))

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
            "predicted_duration": 25.598840843668743
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
        pass

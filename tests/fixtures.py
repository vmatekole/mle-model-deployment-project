import pytest
import pandas as pd


@pytest.fixture(scope='session')
def taxi_rides() -> list[dict]:
    return [
        {
            "ride_id": "Ride-1",
            "PULocationID": 1,
            "DOLocationID": 2,
            "trip_distance": 2.5
        },
        {
            "ride_id": "Ride-2",
            "PULocationID": 2,
            "DOLocationID": 3,
            "trip_distance": 5.0
        },
        {
            "ride_id": "Ride-3",
            "PULocationID": 3,
            "DOLocationID": 4,
            "trip_distance": 7.5
        },
        {
            "ride_id": "Ride-4",
            "PULocationID": 4,
            "DOLocationID": 5,
            "trip_distance": 10.0
        },
        {
            "ride_id": "Ride-5",
            "PULocationID": 5,
            "DOLocationID": 6,
            "trip_distance": 12.5
        },
        {
            "ride_id": "Ride-6",
            "PULocationID": 6,
            "DOLocationID": 7,
            "trip_distance": 15.0
        },
        {
            "ride_id": "Ride-7",
            "PULocationID": 7,
            "DOLocationID": 8,
            "trip_distance": 17.5
        }
    ]


@pytest.fixture(scope='session')
def expected_predictions():
    return [
        {
            "ride_id": "Ride-1",
            "PULocationID": 1,
            "DOLocationID": 2,
            "trip_distance": 2.5,
            "predicted_duration": 25.598840843668743
        },
        {
            "ride_id": "Ride-2",
            "PULocationID": 2,
            "DOLocationID": 3,
            "trip_distance": 5,
            "predicted_duration": 25.5989543361311
        },
        {
            "ride_id": "Ride-3",
            "PULocationID": 3,
            "DOLocationID": 4,
            "trip_distance": 7.5,
            "predicted_duration": 25.599067828593462
        },
        {
            "ride_id": "Ride-4",
            "PULocationID": 4,
            "DOLocationID": 5,
            "trip_distance": 10,
            "predicted_duration": 25.599181321055823
        },
        {
            "ride_id": "Ride-5",
            "PULocationID": 5,
            "DOLocationID": 6,
            "trip_distance": 12.5,
            "predicted_duration": 25.59929481351818
        },
        {
            "ride_id": "Ride-6",
            "PULocationID": 6,
            "DOLocationID": 7,
            "trip_distance": 15,
            "predicted_duration": 25.599408305980543
        },
        {
            "ride_id": "Ride-7",
            "PULocationID": 7,
            "DOLocationID": 8,
            "trip_distance": 17.5,
            "predicted_duration": 25.5995217984429
        }
    ]
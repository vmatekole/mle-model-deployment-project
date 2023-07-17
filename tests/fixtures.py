import pytest

DURATION_TOLERANCE=0.01

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
            "predicted_duration": pytest.approx(12.745262840390389, abs=DURATION_TOLERANCE)
        },
        {
            "ride_id": "Ride-2",
            "PULocationID": 2,
            "DOLocationID": 3,
            "trip_distance": 5.0,
            "predicted_duration": pytest.approx(20.459175143160792, abs=DURATION_TOLERANCE)
        },
        {
            "ride_id": "Ride-3",
            "PULocationID": 3,
            "DOLocationID": 4,
            "trip_distance": 7.5,
            "predicted_duration": pytest.approx(24.36308116213256, abs=DURATION_TOLERANCE)
        },
        {
            "ride_id": "Ride-4",
            "PULocationID": 4,
            "DOLocationID": 5,
            "trip_distance": 10.0,
            "predicted_duration": pytest.approx(27.405308446387867, abs=DURATION_TOLERANCE)
        },
        {
            "ride_id": "Ride-5",
            "PULocationID": 5,
            "DOLocationID": 6,
            "trip_distance": 12.5,
            "predicted_duration": pytest.approx(33.34291453535095,abs=DURATION_TOLERANCE)
        },
        {
            "ride_id": "Ride-6",
            "PULocationID": 6,
            "DOLocationID": 7,
            "trip_distance": 15.0,
            "predicted_duration": pytest.approx(34.443857854731235, abs=DURATION_TOLERANCE)
        },
        {
            "ride_id": "Ride-7",
            "PULocationID": 7,
            "DOLocationID": 8,
            "trip_distance": 17.5,
            "predicted_duration": pytest.approx(34.443857854731235, abs=DURATION_TOLERANCE)
        }
    ]
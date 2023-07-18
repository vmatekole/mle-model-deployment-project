import numpy as np
import pytest
from data_model import TaxiRide
from main import MODEL
from predict import prepare_features
from rich import print
from rich.console import Console


class TestDirectionalBehaviour:

    @pytest.mark.parametrize('short_ride, long_ride', [(
        TaxiRide(**{
            'ride_id': 'Ride-1',
            'PULocationID': 1,
            'DOLocationID': 2,
            'trip_distance': 2.5
        }),
        TaxiRide(**{
            "ride_id": "Ride-5",
            "PULocationID": 5,
            "DOLocationID": 7,
            'trip_distance': 15
        }))])
    def test_direction(self, long_ride: dict[str, int, int], short_ride: dict[str, int, int]):

        directional_examples = [prepare_features(ride) for ride in [
            long_ride, short_ride]]

        directional_predictions = MODEL.predict(directional_examples)
        # Test case 1 check that predicted duration is longer for short rides vs long rides
        assert np.where(
            directional_predictions[0] > directional_predictions[1])

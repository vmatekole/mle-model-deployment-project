import mlflow


def prepare_features(ride):
    features = {}
    features['trip_route'] = f"{ride.PULocationID}_{ride.DOLocationID}"
    features['trip_distance'] = ride.trip_distance
    return features

def predict(model, uri, data):
    mlflow.set_tracking_uri(uri)
    model_input = prepare_features(data)
    prediction = model.predict(model_input)
    return float(prediction[0])
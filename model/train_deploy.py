import os
import traceback
from typing import Tuple

import mlflow
import optuna
import pandas as pd
from dotenv import load_dotenv
from rich import print
from rich.console import Console
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline

import logging

YEAR = 2021
MONTH = 1
COLOUR = "yellow"

MLFLOW = mlflow

FEATURES = ['PULocationID', 'DOLocationID', 'trip_distance']
TARGET = 'duration'


def load_data() -> pd.DataFrame:
    if not os.path.exists(f"./data/{COLOUR}_tripdata_{YEAR}-{MONTH:02d}.parquet"):
        os.system(
            f"wget -P ./data https://d37ci6vzurychx.cloudfront.net/trip-data/{COLOUR}_tripdata_{YEAR}-{MONTH:02d}.parquet")

    df = pd.read_parquet(
        f"./data/{COLOUR}_tripdata_{YEAR}-{MONTH:02d}.parquet")
    return df


def calculate_trip_duration_in_minutes(df):
    df['trip_duration_minutes'] = (
        df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    df = df[(df['trip_duration_minutes'] >= 1) &
            (df['trip_duration_minutes'] <= 60)]

    return df


def preprocess(df_raw) -> pd.DataFrame:
    df = df_raw.copy()
    df = calculate_trip_duration_in_minutes(df)

    categorical_features = ['PULocationID', 'DOLocationID']
    df[categorical_features] = df[categorical_features].astype(str)

    df['trip_route'] = df['PULocationID'] + "_" + df['DOLocationID']
    df = df[['trip_route', 'trip_distance', 'trip_duration_minutes']]

    return df


def train(model, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame) -> float:

    X_train = X_train.to_dict(orient="records")
    X_test = X_test.to_dict(orient="records")

    # Setup the MLflow experiment
    # exp_name = 'yellow-taxi-trip-duration'
    # MLFLOW.set_experiment(exp_name)

    # with MLFLOW.start_run():
    #     tags = {
    #         "model": "Random Forest Regressor",
    #         "developer": "Victor Matekole",
    #         "dataset": f"{COLOUR}-taxi",
    #         "year": YEAR,
    #         "month": MONTH,
    #         "features": FEATURES,
    #         "target": TARGET
    #     }
    #     MLFLOW.set_tags(tags)

    pipeline = make_pipeline(
        DictVectorizer(),
        model
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    MLFLOW.log_metric('rmse', rmse)

    MLFLOW.sklearn.log_model(pipeline, "model")
    return rmse

def init():
    load_dotenv()

    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
    # Set up the connection to MLflow
    MLFLOW.set_tracking_uri(MLFLOW_TRACKING_URI)

    SA_KEY = os.getenv("SA_KEY")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SA_KEY


def train_test_sets(df_processed: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df_processed['trip_duration_minutes']
    X = df_processed.drop(columns=['trip_duration_minutes'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2)

    return X_train, X_test, y_train, y_test

# Define the objective function for optimization


def objective(trial, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame):
    # Define the hyperparameters to optimize

    with MLFLOW.start_run():
        tags = {
            'model': 'Random Forest Regressor',
            'developer': 'Victor Matekole',
            'dataset': f'{COLOUR}-taxi',
            'year': YEAR,
            'month': MONTH,
            'features': FEATURES,
            'target': TARGET
        }
        MLFLOW.set_tags(tags)

        params = {
            'criterion': trial.suggest_categorical('criterion', ['squared_error']),
            'n_estimators': trial.suggest_int('n_estimators', 10, 50),
            'max_depth': trial.suggest_int('max_depth', 1, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 50)
        }
        # MLFLOW.log_params(params)

        # Create a RandomForestRegressor with the suggested hyperparameters
        model = RandomForestRegressor(random_state=42, **params)

        rmse = train(model, X_train, y_train, X_test, y_test)

        # Print the best hyperparameters and best score
        MLFLOW.log_params(params)
        MLFLOW.log_metric('rmse',rmse)
        # Add a description to the best model

        return rmse

        
     


def run_optuna(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame):
    # Setup the MLflow experiment
    exp_name = 'optuna-runs'
    MLFLOW.set_experiment(exp_name)

   
    # Create an Optuna study object
    study = optuna.create_study(direction='minimize')

    # Optimize the objective function
    try:
        study.optimize(lambda trial: objective(
            trial, X_train, y_train, X_test, y_test), n_trials=20)

        # Print the best hyperparameters and best score
        # TODO: Attach best model and put into production
        MLFLOW.log_params(study.best_trial.params)
        MLFLOW.log_metric('rmse',study.best_trial.value)
    except Exception as e:
        traceback.print_exc()


def main() -> None:
    init()
    logging.getLogger("mlflow").setLevel(logging.DEBUG)
    console = Console()
    print('Loading data :smiley: \n')
    df_raw = load_data()
    print(df_raw.head())
    print(df_raw.isnull().sum())

    print('\nProcessing data :smiley: \n')
    df_processed = preprocess(df_raw)
    print(df_processed.head())

    X_train, X_test, y_train, y_test = train_test_sets(df_processed)
    print('Start training')
    # train(RandomForestRegressor(random_state=42,criterion='squared_error', max_depth=1),X_train, y_train, X_test, y_test)

    # console.print('Training completed\n', style='bold green')

    console.print('Running Optuna\n')
    run_optuna(X_train, y_train, X_test, y_test)
    console.print('Optuna completed\n', style='bold green')


main()

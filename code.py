
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import optuna
import mlflow
import mlflow.sklearn

# MLflow Setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Flight_Price_Prediction")


mlflow.set_experiment("Flight Price Prediction - Random Forest Optuna Optimization")

# Load and preprocess data
df = pd.read_csv("clean_dataset.csv")
df = df.drop('Unnamed: 0', axis=1)
df = df.drop('flight', axis=1)

# Feature engineering
df['class'] = df['class'].apply(lambda x: 1 if x == 'Business' else 0)
df.stops = pd.factorize(df.stops)[0]
df = df.join(pd.get_dummies(df.airline, prefix='airline')).drop('airline', axis=1)
df = df.join(pd.get_dummies(df.source_city, prefix='source')).drop('source_city', axis=1)
df = df.join(pd.get_dummies(df.destination_city, prefix='dest')).drop('destination_city', axis=1)
df = df.join(pd.get_dummies(df.arrival_time, prefix='arrival')).drop('arrival_time', axis=1)
df = df.join(pd.get_dummies(df.departure_time, prefix='departure')).drop('departure_time', axis=1)

# Prepare data
X, y = df.drop('price', axis=1), df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optuna objective function
def objective(trial):
    with mlflow.start_run():
        # Hyperparameters
        n_estimators = trial.suggest_int('n_estimators', 100, 500)
        max_depth = trial.suggest_int('max_depth', 10, 50)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

        # Model training
        reg = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1
        )
        reg.fit(X_train, y_train)

        # Evaluation
        y_pred = reg.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        error = np.sqrt(mse)

        # MLflow logging
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features
        })
        mlflow.log_metric("rmse", error)
        mlflow.sklearn.log_model(reg, "random_forest_model")

        return error

# Optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)

# Results
print("Best hyperparameters: ", study.best_params)
print("Best MSE: ", study.best_value)

# Final model
best_params = study.best_params
best_model = RandomForestRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_features=best_params['max_features'],
    random_state=42,
    n_jobs=-1
)
best_model.fit(X_train, y_train)
mlflow.sklearn.log_model(best_model, "best_random_forest_model")

# XGBoost setup (remaining cells)
import mlflow
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
train = xgb.DMatrix(X_train, label=y_train)
valid = xgb.DMatrix(X_val, label=y_val)

def objective(params):
    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(params)
        mlflow.xgboost.autolog()
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )
        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)
    return {'loss': rmse, 'status': STATUS_OK}

mlflow.end_run()

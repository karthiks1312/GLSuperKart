# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# for model serialization
import joblib
# for creating a folder
import os

# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

# -------------------------------
# MLflow setup
# -------------------------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLOps_CICD_experiment")

REPO_ID = os.getenv("HF_REPO_ID")
PROJECT_NAME = os.getenv("REPO_NAME")
DATASET_REPO_ID = f"{REPO_ID}/{PROJECT_NAME}"


HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)


Xtrain = pd.read_csv(hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename="Xtrain.csv",
        repo_type="dataset"
    ))
Xtest  = pd.read_csv(hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename="Xtest.csv",
        repo_type="dataset"
    ))
ytrain = pd.read_csv(hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename="ytrain.csv",
        repo_type="dataset"
    ))
ytest  = pd.read_csv(hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename="ytest.csv",
        repo_type="dataset"
    ))

print("SuperKart train-test data loaded successfully")

# -------------------------------
# Feature groups (SuperKart)
# -------------------------------
numeric_features = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Establishment_Year"
]

categorical_features = [
    "Product_Sugar_Content",
    "Product_Type",
    "Store_Id",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type"
]

# -------------------------------
# Preprocessor
# -------------------------------
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# -------------------------------
# XGBoost Regressor
# -------------------------------
xgb_model = xgb.XGBRegressor(
    random_state=42,
    n_jobs=-1,
    objective="reg:squarederror"
)

# -------------------------------
# Hyperparameter grid
# -------------------------------
param_grid = {
    "xgbregressor__n_estimators": [100, 200],
    "xgbregressor__max_depth": [3, 5, 7],
    "xgbregressor__learning_rate": [0.01, 0.05, 0.1],
    "xgbregressor__subsample": [0.7, 0.8, 1.0],
    "xgbregressor__colsample_bytree": [0.7, 0.8, 1.0],
    "xgbregressor__reg_lambda": [0.1, 1, 10]
}

# -------------------------------
# Pipeline
# -------------------------------
model_pipeline = make_pipeline(preprocessor, xgb_model)

# -------------------------------
# Training + MLflow logging
# -------------------------------
with mlflow.start_run(run_name="SuperKart_XGBoost"):

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring="neg_mean_squared_error"
    )

    grid_search.fit(Xtrain, ytrain.values.ravel())

    # Log all parameter combinations
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_neg_mse", results["mean_test_score"][i])

    # Best model
    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)

    # Predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)


    # Metrics

    metrics = {
    "train_RMSE": mean_squared_error(ytrain, y_pred_train) ** 0.5,
    "test_RMSE": mean_squared_error(ytest, y_pred_test) ** 0.5,
    "train_MAE": mean_absolute_error(ytrain, y_pred_train),
    "test_MAE": mean_absolute_error(ytest, y_pred_test),
    "train_R2": r2_score(ytrain, y_pred_train),
    "test_R2": r2_score(ytest, y_pred_test),
}

    mlflow.log_metrics(metrics)

    # -------------------------------
    # Save model
    # -------------------------------
    model_path = "best_superkart_sales_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")

    print(f"Model saved locally as {model_path}")

    # -------------------------------
    # Upload model to Hugging Face
    # -------------------------------
    REPO_ID = f"{REPO_ID}/superkart-sales-model"
    repo_type = "model"

    try:
        api.repo_info(repo_id=REPO_ID, repo_type=repo_type)
        print(f"Model repo '{REPO_ID}' already exists.")
    except RepositoryNotFoundError:
        print(f"Creating model repo '{REPO_ID}'...")
        create_repo(repo_id=REPO_ID, repo_type=repo_type, private=False)

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=REPO_ID,
        repo_type=repo_type,
    )

    print("Model uploaded to Hugging Face successfully")

# 🛒 GLSuperKart: End-to-End MLOps Sales Forecasting

An end-to-end machine learning project that predicts **total product sales** for a retail chain (SuperKart) from product and store attributes. It covers the full MLOps lifecycle: data versioning, preprocessing, hyperparameter-tuned model training with experiment tracking, and automated deployment of a web app through a CI/CD pipeline.

**Live demo:** [huggingface.co/spaces/karthiks969/GLSuperKart](https://huggingface.co/spaces/karthiks969/GLSuperKart)  
**Model:** [karthiks969/superkart-sales-model](https://huggingface.co/karthiks969/superkart-sales-model) · **Dataset:** [karthiks969/GLSuperKart](https://huggingface.co/datasets/karthiks969/GLSuperKart)

## Highlights

- **Automated CI/CD** with GitHub Actions: one push to `main` runs the whole pipeline (register data → prepare data → train model → deploy app).
- **Data and model versioning** on the Hugging Face Hub (dataset repo and model repo).
- **Experiment tracking** with MLflow. Every hyperparameter combination, the best parameters, and the train/test metrics are logged.
- **XGBoost regression** in a scikit-learn pipeline (`StandardScaler` + `OneHotEncoder`), tuned with `GridSearchCV`.
- **Containerised Streamlit app** (Docker) deployed to a Hugging Face Space.

## Architecture

```
push to main
     │
     ▼
┌──────────────────┐   ┌───────────────┐   ┌──────────────────┐   ┌──────────────────┐
│ register-dataset │──▶│   data-prep   │──▶│  model-training  │──▶│  deploy-hosting  │
│ upload CSV to HF │   │ clean, encode,│   │ GridSearchCV +   │   │ push Docker +    │
│ dataset repo     │   │ train/test    │   │ MLflow logging,  │   │ Streamlit app to │
│                  │   │ split, upload │   │ push model to HF │   │ HF Space         │
└──────────────────┘   └───────────────┘   └──────────────────┘   └──────────────────┘
```

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python 3.10 |
| Modelling | XGBoost, scikit-learn, pandas |
| Experiment tracking | MLflow |
| Versioning and hosting | Hugging Face Hub (datasets, models, Spaces) |
| App | Streamlit |
| Containerisation | Docker |
| CI/CD | GitHub Actions |

## Dataset

`SuperKart.csv` has roughly 8,760 rows of product-level sales across five stores.

- **Product features:** weight, sugar content, allocated display area, type, MRP
- **Store features:** store ID, establishment year, size, city tier, store type
- **Target:** `Product_Store_Sales_Total`

## Project Structure

```
.
├── .github/workflows/pipeline.yml     # CI/CD pipeline
└── GLSuperkart/
    ├── data/SuperKart.csv             # raw dataset
    ├── model_building/
    │   ├── data_register.py           # upload dataset to HF Hub
    │   ├── prep.py                    # preprocessing + train/test split
    │   └── train.py                   # tuning, MLflow tracking, model upload
    ├── hosting/hosting.py             # deploy app to HF Space
    ├── deployment/
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   └── src/streamlit_app.py       # prediction web app
    └── requirements.txt
```

## Model

- **Algorithm:** `XGBRegressor` inside a scikit-learn pipeline
- **Tuning:** 3-fold `GridSearchCV` over `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree` and `reg_lambda`
- **Metrics tracked:** RMSE, MAE and R² on train and test sets

| Metric | Train | Test |
|---|---|---|
| RMSE | 220.23 | 284.59 |
| MAE | 83.77 | 113.09 |
| R² | 0.957 | **0.929** |

The model explains about 93% of the variance in held-out sales (80/20 split, `random_state=42`; average sales are about ₹3,464). Best parameters: `n_estimators=200`, `max_depth=7`, `learning_rate=0.05`, `subsample=0.7`, `colsample_bytree=1.0`, `reg_lambda=10`.

## Running Locally

```bash
git clone <your-repo-url>
cd GLSuperKart
pip install -r GLSuperkart/requirements.txt

# environment variables (also used as GitHub Actions secrets)
export HF_TOKEN=<hugging-face-token>
export HF_REPO_ID=<hugging-face-username>
export REPO_NAME=GLSuperkart

python GLSuperkart/model_building/data_register.py
python GLSuperkart/model_building/prep.py

mlflow ui --port 5000 &          # tracking server used by train.py
python GLSuperkart/model_building/train.py

streamlit run GLSuperkart/deployment/src/streamlit_app.py
```

## CI/CD Secrets

Add these under **Settings → Secrets and variables → Actions**: `HF_TOKEN`, `HF_REPO_ID`, `REPO_NAME`, `PROJECT_NAME`.

## Possible Improvements

- Add automated tests and a model-quality gate before deployment
- Persist MLflow tracking beyond the CI run
- Add data validation and drift monitoring

## Author

**Karthik S**: [GitHub](https://github.com/karthiks969) · [Hugging Face](https://huggingface.co/karthiks969)

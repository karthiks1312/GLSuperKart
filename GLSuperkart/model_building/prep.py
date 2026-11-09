# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# ---------------------------
# Hugging Face Authentication
# ---------------------------

HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)


REPO_ID = os.getenv("HF_REPO_ID")

PROJECT_NAME = os.getenv("REPO_NAME")


# ---------------------------
# Load SuperKart dataset
# ---------------------------
DATASET_PATH = (
    f"hf://datasets/{REPO_ID}/{PROJECT_NAME}/SuperKart.csv"
)

df = pd.read_csv(DATASET_PATH)
print("SuperKart dataset loaded successfully")

# ---------------------------
# Drop identifier column
# ---------------------------
df.drop(columns=["Product_Id"], inplace=True)

# ---------------------------
# Clean categorical values
# (encoding is done by the OneHotEncoder inside the training pipeline,
#  so the app can pass raw category names at inference time)
# ---------------------------
df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace({"reg": "Regular"})

# ---------------------------
# Define target variable
# ---------------------------
target_col = "Product_Store_Sales_Total"

X = df.drop(columns=[target_col])
y = df[target_col]

# ---------------------------
# Train-test split
# ---------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------
# Save splits locally
# ---------------------------
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# ---------------------------
# Upload to Hugging Face dataset repo
# ---------------------------
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file in files:
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file,
        repo_id=f"{REPO_ID}/{PROJECT_NAME}",
        repo_type="dataset",
    )

print("SuperKart train-test splits uploaded successfully")

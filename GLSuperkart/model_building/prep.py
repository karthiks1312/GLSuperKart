# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
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
# Encode categorical columns
# ---------------------------
label_encoder = LabelEncoder()

categorical_cols = [
    "Product_Sugar_Content",
    "Product_Type",
    "Store_Id",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type"
]

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

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

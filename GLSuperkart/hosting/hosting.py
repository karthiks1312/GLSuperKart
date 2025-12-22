from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

REPO_ID = os.getenv("HF_REPO_ID")
PROJECT_NAME = os.getenv("PROJECT_NAME")

HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

REPO_ID = f"{REPO_ID}/GLSuperKart"
repo_type = "space"

# -------------------------------
# Check if Space exists
# -------------------------------
try:
    api.repo_info(repo_id=REPO_ID, repo_type=repo_type)
    print(f"Space '{REPO_ID}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{REPO_ID}' not found. Creating new Space...")
    create_repo(
        repo_id=REPO_ID,
        repo_type=repo_type,
        space_sdk="docker",   # IMPORTANT for Streamlit apps
        private=False
    )
    print(f"Space '{REPO_ID}' created successfully.")

# -------------------------------
# Upload app files
# -------------------------------

upload_path = "GLSuperkart/deployment"

api.upload_folder(
    folder_path=upload_path,
    repo_id=REPO_ID,
    repo_type=repo_type,
    path_in_repo=""
)

print("Deployment files uploaded to Hugging Face Space")

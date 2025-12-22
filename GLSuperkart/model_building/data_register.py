from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

REPO_ID = os.getenv("HF_REPO_ID")
PROJECT_NAME = os.getenv("PROJECT_NAME")

repo_name=f"{PROJECT_NAME}"


repo_Id = f"{REPO_ID}/{repo_name}"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_Id, repo_type=repo_type)
    print(f"Space '{repo_Id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_Id}' not found. Creating new space...")
    create_repo(repo_id=repo_Id, repo_type=repo_type, private=False)
    print(f"Space '{repo_Id}' created.")

path = f"{PROJECT_NAME}/data"

api.upload_folder(
    folder_path=path,
    repo_id=repo_Id,
    repo_type=repo_type,
)

from huggingface_hub import hf_hub_download
import tarfile
import os

# dataset repo
REPO_ID = "yutengz/robotic-manipulation-dataset"
TARGET_DIR = "original_data"

# three .tar.gz
FILES = [
    "block_hammer_beat_sf50_D435_pkl.tar.gz",
    "block_handover_sf50_D435_pkl.tar.gz",
    "blocks_stack_easy_sf50_D435_pkl.tar.gz",
]

os.makedirs(TARGET_DIR, exist_ok=True)

for file_name in FILES:
    # download
    file_path = hf_hub_download(repo_id=REPO_ID, filename=file_name)
    
    # unzip dataset to folder
    with tarfile.open(file_path, "r:gz") as tar:
        print(f"Extracting {file_name}...")
        tar.extractall(path=TARGET_DIR)

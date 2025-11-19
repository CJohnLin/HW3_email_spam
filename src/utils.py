# src/utils.py
import os
from typing import List

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def list_models(models_dir: str = "models"):
    files = []
    if os.path.isdir(models_dir):
        files = [f for f in os.listdir(models_dir) if f.endswith("_model.pkl")]
    return files

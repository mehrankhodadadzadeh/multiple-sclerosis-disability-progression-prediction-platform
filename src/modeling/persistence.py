from pathlib import Path
import json
import joblib
from src.config import write_json


def save_model(pipeline,metadata,model_dir):
    root = Path(model_dir); root.mkdir(parents=True,exist_ok=True)
    write_json(root/'model_metadata.json',metadata)
    # Normalize undefined metrics to JSON null in BOTH metadata representations.
    normalized = json.loads((root/'model_metadata.json').read_text())
    joblib.dump({'pipeline':pipeline,'metadata':normalized},root/'best_model.joblib')


def load_model(model_dir):
    # joblib may execute code: load ONLY trusted locally generated artifacts.
    return joblib.load(Path(model_dir)/'best_model.joblib')

"""CLI paths and integrity checks for the explicit cohort-review workflow."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import load_config


def arguments(description,data=False):
    parser=argparse.ArgumentParser(description=description)
    if data: parser.add_argument('--data-dir',required=True,type=Path)
    for flag,env,default in [('config-dir','MS_CONFIG_DIR','configs'),('report-dir','MS_REPORT_DIR','reports'),('model-dir','MS_MODEL_DIR','models'),('artifact-dir','MS_ARTIFACT_DIR','artifacts')]:
        parser.add_argument('--'+flag,default=os.getenv(env,default),type=Path)
    return parser


def fingerprint(data_dir,config):
    h=hashlib.sha256(json.dumps(config,sort_keys=True).encode())
    for spec in config['data']['tables'].values():
        with (data_dir/spec['filename']).open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def run(main):
    try: main()
    except (ValueError,FileNotFoundError) as exc:
        # Application errors are intentionally aggregate-only.
        raise SystemExit(str(exc)) from None

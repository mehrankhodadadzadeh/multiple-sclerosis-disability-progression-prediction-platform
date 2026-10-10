"""Read only the nine configured processed CSVs; no raw or mapping-file discovery."""
from pathlib import Path
import pandas as pd
from src.data.validation import convert_table


def load_tables(data_dir, config):
    root = Path(data_dir)
    missing = [s['filename'] for s in config['tables'].values() if not (root / s['filename']).is_file()]
    if missing: raise ValueError(f'Missing required CSV files: {missing}')
    tables, reports = {}, {}
    for name, spec in config['tables'].items():
        raw = pd.read_csv(root / spec['filename'], dtype='string')
        tables[name], reports[name] = convert_table(raw, spec, config, name)
    return tables, reports

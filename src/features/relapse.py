import numpy as np
import pandas as pd
from src.features.edss import history


def extract(frame,index_date,config):
    columns = [c for c in ('start_date','type','inn','corticosteroid') if c in frame]
    data = history(frame,index_date,'start_date')[columns].drop_duplicates()
    result = {'days_since_last_relapse_therapy': (index_date-data.iloc[-1].start_date).days if len(data) else np.nan}
    for months in [6,12,24]:
        recent = data[data.start_date >= index_date-pd.DateOffset(months=months)]
        result[f'relapse_therapy_count_{months}m'] = len(recent)
        if months != 6:
            result[f'corticosteroid_episode_count_{months}m'] = int(recent.corticosteroid.eq(1).sum()) if recent.corticosteroid.notna().all() else np.nan
    return result

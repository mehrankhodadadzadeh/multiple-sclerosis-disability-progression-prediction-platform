import numpy as np
import pandas as pd
from src.features.edss import history
from src.data.validation import invalid_treatment_intervals


def normalize_inn(values):
    """Minimal label normalization only; no brand/generic mapping."""
    return values.astype('string').str.strip().str.upper().fillna('UNKNOWN').replace('', 'UNKNOWN')


def extract(frame,index_date,config):
    # Exclude impossible episodes before masking future ends; retain all valid episodes.
    # Do not use exported status/latest_date/cumulative dose/stop reason.
    valid = frame.loc[~invalid_treatment_intervals(frame)]
    data = history(valid,index_date,'start_date').copy()
    data['inn'] = normalize_inn(data.inn)
    data.loc[data.end_date > index_date,'end_date'] = pd.NaT
    active = data[data.end_date.isna()]
    count = len(active)
    current = active.iloc[0] if count == 1 else None
    ended = data[data.end_date.notna()]
    return {'currently_on_dmt': int(count > 0),
            'active_dmt_count': count,
            'current_dmt': current.inn if current is not None else ('MULTIPLE' if count > 1 else 'NONE'),
            'current_dmt_duration_days': (index_date-current.start_date).days if current is not None else np.nan,
            'number_previous_dmts': len(ended),
            'distinct_previous_dmts': int(ended.loc[ended.inn.ne('UNKNOWN'), 'inn'].nunique()),
            'dmt_starts_previous_24m': int((data.start_date >= index_date-pd.DateOffset(months=24)).sum()),
            'days_since_last_dmt_start': (index_date-data.iloc[-1].start_date).days if len(data) else np.nan}

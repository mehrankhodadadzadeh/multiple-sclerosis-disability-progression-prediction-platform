import numpy as np
import pandas as pd
from src.features.edss import history, series_history, change, slope
from src.data.validation import reconcile_same_day


def extract(frame, index_date, config):
    frame, _ = reconcile_same_day(history(frame, index_date))
    data = series_history(frame,index_date,'sdmt')
    recent = data[data.date >= index_date-pd.DateOffset(months=config['cohort']['lookback_months'])]
    result = {'latest_sdmt': float(data.iloc[-1].sdmt) if len(data) else np.nan,
              'sdmt_slope_24m': slope(recent,'sdmt'), 'sdmt_count_24m': len(recent),
              'days_since_last_sdmt': (index_date-data.iloc[-1].date).days if len(data) else np.nan}
    for months in [12,24]: result[f'sdmt_change_{months}m'] = change(data,index_date,'sdmt',months,config['features']['change_reference_tolerance_months'])
    return result

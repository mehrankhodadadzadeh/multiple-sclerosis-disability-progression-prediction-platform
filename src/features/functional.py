import numpy as np
import pandas as pd
from src.features.edss import history, series_history, change, slope
from src.data.validation import reconcile_same_day


def extract(frame,index_date,config):
    frame, _ = reconcile_same_day(history(frame, index_date))
    result = {}
    for column, latest_name, change_name in [('t25fw_mean','latest_t25fw','t25fw'),('nine_hpt_right_mean','latest_9hpt_right','nine_hpt_right'),('nine_hpt_left_mean','latest_9hpt_left','nine_hpt_left'),('pasat3','latest_pasat','pasat')]:
        if column != 'pasat3' and config['features']['nonpositive_timed_tests_as_missing']:
            frame.loc[frame[column] <= 0, column] = np.nan
        if column == 'pasat3':
            # Clean only this historical feature copy; loaded PASAT stays unchanged.
            minimum = 1 if config['features']['pasat_zero_as_missing'] else 0
            frame.loc[~frame[column].between(minimum, 60),column] = np.nan
        data = series_history(frame,index_date,column)
        result[latest_name] = float(data.iloc[-1][column]) if len(data) else np.nan
        result[f'{change_name}_change_24m'] = change(data,index_date,column,24,config['features']['change_reference_tolerance_months'])
        if column == 't25fw_mean':
            recent = data[data.date >= index_date-pd.DateOffset(months=config['cohort']['lookback_months'])]
            result['t25fw_slope_24m'] = slope(recent,column)
    return result

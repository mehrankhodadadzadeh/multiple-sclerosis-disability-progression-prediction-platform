"""Historical measurement helpers shared by EDSS, cognition and functional tests."""
import numpy as np
import pandas as pd


def history(frame, index_date, date_col='date'):
    return frame[frame[date_col].notna() & (frame[date_col] <= index_date)].sort_values(date_col)


def series_history(frame, index_date, column):
    data = history(frame, index_date).dropna(subset=[column])[['date',column]].drop_duplicates()
    if data.date.duplicated().any(): raise ValueError('Conflicting same-day clinical measurements; resolve before modeling')
    return data


def change(data, index_date, column, months, tolerance):
    if data.empty: return np.nan
    reference = index_date - pd.DateOffset(months=months)
    earlier = data[(data.date <= reference) & (data.date >= reference - pd.DateOffset(months=tolerance))]
    # Do not compare a stale latest test to itself.
    if earlier.empty or data.iloc[-1].date <= reference: return np.nan
    return float(data.iloc[-1][column] - earlier.iloc[-1][column])


def slope(data, column):
    if len(data) < 2: return np.nan
    x = (data.date - data.date.min()).dt.days.to_numpy() / 365.25
    return float(np.polyfit(x, data[column].to_numpy(), 1)[0]) if np.ptp(x) else np.nan


def extract(frame, index_date, config):
    data = series_history(frame, index_date, 'edss')
    recent = data[data.date >= index_date - pd.DateOffset(months=config['cohort']['lookback_months'])]
    previous = data[data.date < index_date]
    values = recent.edss
    result = {'current_edss': float(data.iloc[-1].edss) if len(data) else np.nan,
              'previous_edss': float(previous.iloc[-1].edss) if len(previous) else np.nan,
              'days_since_previous_edss': (index_date - previous.iloc[-1].date).days if len(previous) else np.nan,
              'edss_slope_24m': slope(recent, 'edss'), 'edss_count_24m': len(recent)}
    for name in ['mean','min','max','std']: result[f'edss_{name}_24m'] = float(getattr(values, name)()) if len(values) else np.nan
    for months in [12,24]: result[f'edss_change_{months}m'] = change(data,index_date,'edss',months,config['features']['change_reference_tolerance_months'])
    return result

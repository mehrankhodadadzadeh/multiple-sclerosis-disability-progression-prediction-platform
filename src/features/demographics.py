import numpy as np
import pandas as pd


def extract(patient, index_date, config):
    birth, onset = patient.date_of_birth, patient.ms_onset_date
    onset_known = pd.notna(onset) and onset <= index_date
    return {'age_at_index': (index_date-birth).days/365.25 if pd.notna(birth) else np.nan,
            'sex': str(patient.sex) if pd.notna(patient.sex) else np.nan,
            'age_at_ms_onset': (onset-birth).days/365.25 if onset_known and pd.notna(birth) else np.nan,
            'disease_duration_years': (index_date-onset).days/365.25 if onset_known else np.nan,
            'smoking_status': np.nan}

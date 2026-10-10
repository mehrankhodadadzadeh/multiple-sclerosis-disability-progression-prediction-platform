"""Historical eligibility and greedy spacing, independent of future outcomes."""
import pandas as pd


def candidate_indices(edss, patients, config):
    clean = edss.dropna(subset=['date','edss']).drop_duplicates(['study_id','date','edss'])
    if clean.duplicated(['study_id','date']).any(): raise ValueError('Resolve conflicting same-day EDSS measurements')
    records, candidate_count = [], len(clean)
    lookup = patients.set_index('study_id')
    for patient_id, visits in clean.groupby('study_id', sort=True):
        patient = lookup.loc[patient_id]
        last = None
        for visit in visits.sort_values('date').itertuples():
            if pd.isna(patient.date_of_birth): continue
            age = (visit.date - patient.date_of_birth).days / 365.25
            if age < config['minimum_age_years']: continue
            if last is not None and visit.date < last + pd.DateOffset(months=config['minimum_index_spacing_months']): continue
            records.append({'study_id': patient_id, 'index_date': visit.date, 'current_edss': visit.edss})
            last = visit.date
    return pd.DataFrame(records, columns=['study_id','index_date','current_edss']), candidate_count

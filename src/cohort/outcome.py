"""Conservative observed, sustained EDSS progression; this is NOT PIRA."""
import pandas as pd


def classify_outcome(edss, index_date, baseline, horizon_months, config):
    t = pd.Timestamp(index_date)
    horizon = t + pd.DateOffset(months=horizon_months)
    future = edss.loc[(edss.date > t) & edss.edss.notna(), ['date','edss']].sort_values('date').drop_duplicates()
    if future.date.duplicated().any(): raise ValueError('Conflicting same-day EDSS values')
    rule = next((r for r in config['baseline_thresholds'] if r['min_inclusive'] <= baseline < r['max_exclusive']), None)
    if rule is None: raise ValueError('Baseline EDSS outside configured thresholds')
    threshold = baseline + rule['increase']
    within = future[future.date <= horizon]
    unresolved = False
    for row in within.itertuples():
        if row.edss < threshold: continue
        earliest = row.date + pd.DateOffset(months=config['confirmation_min_months'])
        latest = row.date + pd.DateOffset(months=config['confirmation_max_months'])
        subsequent = within[(within.date > row.date) & (within.date <= latest)]
        recovery = subsequent[subsequent.edss < threshold]
        confirmations = subsequent[(subsequent.date >= earliest) & (subsequent.edss >= threshold)]
        # An intervening drop invalidates this episode; a later rise starts anew.
        if not confirmations.empty:
            confirmed_at = confirmations.iloc[0].date
            if recovery.empty or recovery.iloc[0].date > confirmed_at:
                return {'target': 1, 'outcome_status': 'positive', 'outcome_reason': 'sustained_confirmed_worsening'}
        # A recovery resolves the episode; otherwise missing a qualifying check is unknown.
        if recovery.empty: unresolved = True
    if unresolved:
        return {'target': None, 'outcome_status': 'censored', 'outcome_reason': 'unresolved_worsening'}
    grace_end = horizon + pd.DateOffset(months=config['negative_followup_grace_months'])
    boundary = future[(future.date >= horizon) & (future.date <= grace_end)]
    if boundary.empty:
        return {'target': None, 'outcome_status': 'censored', 'outcome_reason': 'no_horizon_assessment'}
    if boundary.iloc[0].date > horizon and boundary.iloc[0].edss >= threshold:
        return {'target': None, 'outcome_status': 'censored', 'outcome_reason': 'boundary_worsening_timing_uncertain'}
    # Require sufficient observed coverage, not just a distant final encounter.
    endpoint = boundary.iloc[0].date
    previous = t
    for date in future.loc[future.date <= endpoint, 'date']:
        if date > previous + pd.DateOffset(months=config['negative_max_observation_gap_months']):
            return {'target': None, 'outcome_status': 'censored', 'outcome_reason': 'sparse_followup'}
        previous = date
    return {'target': 0, 'outcome_status': 'negative', 'outcome_reason': 'adequate_observed_followup'}

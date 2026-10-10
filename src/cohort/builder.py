from pathlib import Path
import pandas as pd
from src.config import write_json
from src.cohort.index_dates import candidate_indices
from src.cohort.outcome import classify_outcome
from src.features import demographics, edss, cognition, functional, treatment, relapse


def build_cohort(tables, config, report_dir=None):
    indices, count = candidate_indices(tables['edss'],tables['patients'],config['cohort'])
    lookup = tables['patients'].set_index('study_id')
    names = ['edss','sdmt','clinical_tests','treatments','relapse_therapy']
    groups = {name: {sid: df for sid,df in tables[name].groupby('study_id')} for name in names}
    records = []
    for index in indices.itertuples(index=False):
        sid, t, baseline = index.study_id, index.index_date, index.current_edss
        data = {name: groups[name].get(sid,tables[name].iloc[:0]) for name in names}
        features = demographics.extract(lookup.loc[sid],t,config)
        for name, module in [('edss',edss),('sdmt',cognition),('clinical_tests',functional),('treatments',treatment),('relapse_therapy',relapse)]:
            features.update(module.extract(data[name],t,config))
        outcome = classify_outcome(data['edss'],t,baseline,config['cohort']['prediction_horizon_months'],config['outcome'])
        records.append({'study_id':sid,'index_date':t,**features,**outcome})
    cohort = pd.DataFrame(records)
    if cohort.empty: raise ValueError('No eligible historical index dates; inspect audit and cohort criteria')
    labels = cohort.target.dropna()
    report = {'candidate_index_dates':count,'eligible_index_dates':len(cohort), 'unique_patients':int(cohort.study_id.nunique()),
              'progression_positive_samples':int(labels.eq(1).sum()),'progression_negative_samples':int(labels.eq(0).sum()),
              'index_dates_with_multiple_active_dmts': int(cohort.active_dmt_count.gt(1).sum()),
              'censored_samples':int(cohort.target.isna().sum()),'binary_training_samples':len(labels),
              'progression_prevalence':float(labels.mean()) if len(labels) else None,
              'outcome_reasons':cohort.outcome_reason.value_counts().to_dict(),
              'feature_missingness':cohort.drop(columns=['study_id','index_date','outcome_reason','outcome_status','target']).isna().mean().to_dict(),
              'exact_duplicate_source_rows':{name:int(df.duplicated().sum()) for name,df in tables.items()},
              'edss_rows_missing_score':int(tables['edss'].edss.isna().sum()),
              'configuration':config}
    if report_dir: write_json(Path(report_dir)/'cohort_summary.json',report)
    return cohort, report

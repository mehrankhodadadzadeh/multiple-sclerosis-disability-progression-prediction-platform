"""Aggregate audit, including missingness and calendar coverage. No patient rows."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.config import write_json
from src.features.treatment import normalize_inn
from src.data.validation import invalid_treatment_intervals, reconcile_same_day


def audit_tables(tables, parsing, config, report_dir):
    report_dir = Path(report_dir); report_dir.mkdir(parents=True, exist_ok=True)
    output = {}
    patients = tables['patients']
    dob = patients.drop_duplicates('study_id').set_index('study_id')['date_of_birth']
    maximum = pd.Timestamp.today().normalize() if config['maximum_date'] == 'today' else pd.Timestamp(config['maximum_date'])
    for name, df in tables.items():
        info = dict(parsing.get(name, {}))
        info.update(rows=len(df), unique_patients=int(df.study_id.nunique()), missing_ids=int(df.study_id.isna().sum()),
                    orphan_ids=int((df.study_id.notna() & ~df.study_id.isin(patients.study_id)).sum()),
                    missingness_by_variable=df.isna().mean().to_dict(), missingness_table=float(df.isna().to_numpy().mean()) if len(df) else None,
                    unexpected_categories={}, range_violations={}, dates_before_birth={}, dates_in_future={})
        # Aggregate distribution only; patient identifiers never leave this routine.
        per_patient = df.drop(columns='study_id').isna().assign(study_id=df.study_id).groupby('study_id').mean().mean(axis=1)
        info['missingness_per_patient_distribution'] = per_patient.describe().to_dict()
        dates = config['tables'][name]['dates']
        for col in dates:
            info['dates_before_birth'][col] = int((df[col] < df.study_id.map(dob)).sum())
            info['dates_in_future'][col] = int((df[col] > maximum).sum())
        for col, allowed in config['categories'].get(name, {}).items():
            info['unexpected_categories'][col] = int((df[col].notna() & ~df[col].isin(allowed)).sum())
        for col, (lo, hi) in config['ranges'].get(name, {}).items():
            invalid = df[col].notna() & ((df[col] < lo) if lo is not None else False)
            if hi is not None: invalid |= df[col] > hi
            if name == 'eq5d': invalid |= df[col].notna() & (df[col] % 1 != 0)
            info['range_violations'][col] = int(invalid.sum())
        info['categorical_distinct_counts'] = {c: int(df[c].nunique()) for c in config['tables'][name]['strings'] if c != 'study_id'}
        info['categorical_fields_without_codebook'] = [c for c in config['tables'][name]['strings'] if c != 'study_id' and c not in config['categories'].get(name, {})]
        info['numeric_distributions'] = {c: df[c].describe().to_dict() for c in config['tables'][name]['numbers']}
        if name == 'patients': info['duplicate_patient_ids'] = int(df.study_id.duplicated().sum())
        if name == 'edss': info['non_half_step_edss'] = int((df.edss.notna() & (df.edss * 2 % 1 != 0)).sum())
        if name == 'clinical_tests':
            info['nonpositive_timed_tests'] = {c: int((df[c] <= 0).sum()) for c in df if c.startswith(('t25fw', 'nine_hpt'))}
            info['pasat_zero'] = int(df.pasat3.eq(0).sum())
            info['pasat_zero_as_missing_count'] = int(df.pasat3.eq(0).sum())
            info['pasat_above_60_count'] = int(df.pasat3.gt(60).sum())
            info['pasat_negative_count'] = int(df.pasat3.lt(0).sum())
            if df.pasat3.notna().any() and (~df.pasat3.dropna().between(1, 60)).any():
                info['handled_quality_warnings'] = ['PASAT outside 1–60 is treated as missing in the modeling view; original values retained.']
        if name in ('sdmt', 'clinical_tests'):
            _, reconciliation = reconcile_same_day(df)
            info.update({f'{name}_{key}': value for key, value in reconciliation.items()})
            if reconciliation['conflicting_patient_date_groups']:
                info.setdefault('handled_quality_warnings', []).append(
                    'Same-day conflicting variables become missing only in the modeling view; other visit measurements and source rows are retained.')
        if 'start_date' in df:
            info['end_before_start'] = int((df.end_date < df.start_date).sum())
            if name == 'treatments':
                invalid = invalid_treatment_intervals(df)
                info['invalid_treatment_intervals_count'] = int(invalid.sum())
                info['excluded_treatment_intervals_count'] = int(invalid.sum())
                if invalid.any():
                    info['handled_quality_warnings'] = ['Invalid treatment intervals excluded from temporal DMT calculations; original rows and dates retained.']
                patients_with_overlap = pairs = same_inn = overlapping_episodes = multiple_indices = 0
                candidate_dates = {sid: group.date.dropna().unique() for sid, group in
                                   tables['edss'].dropna(subset=['edss']).groupby('study_id')}
                for sid, group in df.loc[~invalid].dropna(subset=['start_date']).groupby('study_id'):
                    group = group.sort_values('start_date').copy()
                    group['inn'] = normalize_inn(group.inn)
                    episodes = list(group.itertuples())
                    affected = False
                    for i, current in enumerate(episodes):
                        current_end = current.end_date if pd.notna(current.end_date) else pd.Timestamp.max
                        overlaps_earlier = False
                        for previous in episodes[:i]:
                            previous_end = previous.end_date if pd.notna(previous.end_date) else pd.Timestamp.max
                            if current.start_date < min(current_end, previous_end):
                                pairs += 1
                                same_inn += int(current.inn == previous.inn)
                                affected = overlaps_earlier = True
                        overlapping_episodes += int(overlaps_earlier)
                    patients_with_overlap += int(affected)
                    for date in candidate_dates.get(sid, []):
                        active = (group.start_date <= date) & (group.end_date.isna() | (group.end_date > date))
                        multiple_indices += int(active.sum() > 1)
                info.update(patients_with_overlapping_dmt_episodes=patients_with_overlap,
                            overlapping_dmt_episode_pairs=pairs, same_inn_overlap_pairs=same_inn,
                            different_inn_overlap_pairs=pairs-same_inn,
                            overlapping_treatment_episodes=overlapping_episodes,
                            index_dates_with_multiple_active_dmts=multiple_indices,
                            index_date_basis='Unique EDSS candidate dates before age/spacing eligibility; eligible count is in cohort_summary.json')
                if pairs:
                    info.setdefault('handled_quality_warnings', []).append(
                        'Overlapping valid DMT episodes retained; multiple active episodes use MULTIPLE and missing duration, not inferred switches.')
        date_col = 'date' if 'date' in df else ('start_date' if 'start_date' in df else 'first_exam_date')
        info['duplicate_patient_dates'] = int(df.duplicated(['study_id', date_col]).sum())
        annual = df.assign(year=df[date_col].dt.year).groupby('year').agg(rows=('study_id','size'), patients=('study_id','nunique'))
        info['calendar_year_availability'] = annual.to_dict('index')
        info['missingness_by_calendar_year'] = df.isna().assign(year=df[date_col].dt.year).groupby('year').mean().to_dict('index')
        annual.to_csv(report_dir / f'{name}_annual_counts.csv')
        if not annual.empty:
            fig, ax = plt.subplots(figsize=(7,3)); annual.plot(ax=ax)
            ax.set_title(f'{name}: annual records and patients'); fig.tight_layout()
            fig.savefig(report_dir / f'{name}_annual_counts.png'); plt.close(fig)
        output[name] = info
    # DMT exposure at each year end, distinct from treatment initiations above.
    tr = tables['treatments']
    tr = tr.loc[~invalid_treatment_intervals(tr)].dropna(subset=['start_date'])
    exposure = []
    if not tr.empty:
        for year in range(int(tr.start_date.dt.year.min()), maximum.year + 1):
            date = pd.Timestamp(year, 12, 31)
            active = tr[(tr.start_date <= date) & (tr.end_date.isna() | (tr.end_date > date))]
            exposure.append({'year': year, 'patients_on_dmt': active.study_id.nunique()})
    if exposure:
        pd.DataFrame(exposure).to_csv(report_dir / 'dmt_year_end_exposure.csv', index=False)
    write_json(report_dir / 'data_quality.json', output)
    return output


def ensure_modeling_quality(tables, parsing, config):
    """Fail closed on unusable chronology/core values; audit retains all source rows."""
    issues = []
    core = ('patients','edss','sdmt','clinical_tests','treatments','relapse_therapy')
    for name in core:
        df = tables[name]
        if df.study_id.isna().any(): issues.append(f'{name}: missing study_id')
        if not df.study_id.isin(tables['patients'].study_id).all(): issues.append(f'{name}: orphan study_id')
        for col, stats in parsing[name]['dates'].items():
            if stats['invalid']: issues.append(f'{name}: invalid dates in {col}')
        if any(parsing[name]['invalid_numeric'].values()): issues.append(f'{name}: invalid numeric values')
        key = 'date' if 'date' in df else 'start_date' if 'start_date' in df else None
        if key and df[key].isna().any(): issues.append(f'{name}: missing measurement/start dates')
        if name != 'treatments' and 'end_date' in df and (df.end_date < df.start_date).any(): issues.append(f'{name}: end before start')
        for col, (lo, hi) in config['ranges'].get(name, {}).items():
            if name == 'clinical_tests' and col == 'pasat3':
                continue  # Handled by feature cleaning; retain audit range flags.
            if ((df[col] < lo) if lo is not None else pd.Series(False, index=df.index)).any() or (hi is not None and (df[col] > hi).any()):
                issues.append(f'{name}: values outside configured range in {col}')
        if key:
            dob = tables['patients'].drop_duplicates('study_id').set_index('study_id').date_of_birth
            if (df[key] < df.study_id.map(dob)).any(): issues.append(f'{name}: measurement/start before birth')
            if (df[key] > pd.Timestamp.today().normalize()).any(): issues.append(f'{name}: future measurement/start date')
    p = tables['patients']
    if p.study_id.duplicated().any(): issues.append('patients: duplicate identifiers')
    if (p.ms_onset_date < p.date_of_birth).any(): issues.append('patients: onset before birth')
    if (p.date_of_birth > pd.Timestamp.today().normalize()).any(): issues.append('patients: future birth date')
    if tables['edss'].edss.dropna().mul(2).mod(1).ne(0).any(): issues.append('edss: non half-step values')
    if issues: raise ValueError('Resolve data-quality problems and rerun audit: ' + '; '.join(issues))

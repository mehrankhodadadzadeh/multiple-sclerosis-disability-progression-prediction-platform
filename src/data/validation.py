"""Aggregate-only validation. Never include identifiers or row contents in errors."""
import re
import numpy as np
import pandas as pd


def validate_columns(frame, spec, table):
    required = set(spec['strings'] + spec['dates'] + spec['numbers'])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f'{table}: missing required columns: {missing}')


def parse_dates(series, dayfirst=False, pivot=30):
    """Explicit two-digit-year policy; ambiguous slash dates follow configured order."""
    values = series.astype('string').str.strip()
    ambiguous = values.str.match(r'^\d{1,2}/\d{1,2}/\d{2,4}$', na=False)
    parts = values.str.extract(r'^(\d{1,2})/(\d{1,2})/\d{2,4}$').apply(pd.to_numeric, errors='coerce')
    ambiguous &= parts[0].between(1, 12) & parts[1].between(1, 12) & parts[0].ne(parts[1])
    def expand(value):
        if pd.isna(value): return value
        match = re.fullmatch(r'(\d{1,2}/\d{1,2}/)(\d{2})', value)
        if match:
            y = int(match[2]); return match[1] + str((2000 if y <= pivot else 1900) + y)
        return value
    parsed = pd.to_datetime(values.map(expand), format='mixed', dayfirst=dayfirst, errors='coerce', utc=True).dt.tz_convert(None).dt.normalize()
    return parsed, {'invalid': int((values.notna() & parsed.isna()).sum()), 'ambiguous': int(ambiguous.sum())}


def invalid_treatment_intervals(frame):
    """Flag impossible intervals without repairing or overwriting source dates."""
    return frame.start_date.notna() & frame.end_date.notna() & (frame.end_date < frame.start_date)


def convert_table(raw, spec, config, table):
    validate_columns(raw, spec, table)
    frame = raw.copy()
    report = {'rows': len(raw), 'duplicate_rows': int(raw.duplicated().sum()), 'dates': {}, 'invalid_numeric': {}}
    for col in spec['strings']:
        frame[col] = frame[col].astype('string').str.strip().replace('', pd.NA)
    for col in spec['dates']:
        frame[col], report['dates'][col] = parse_dates(raw[col], config['date_dayfirst'], config['two_digit_year_pivot'])
    for col in spec['numbers']:
        source = raw[col].replace({'true': '1', 'false': '0', 'True': '1', 'False': '0'}) if col in ('deceased', 'corticosteroid') else raw[col]
        numeric = pd.to_numeric(source, errors='coerce').astype(float)
        invalid = raw[col].notna() & (~np.isfinite(numeric))
        report['invalid_numeric'][col] = int(invalid.sum())
        frame[col] = numeric.where(np.isfinite(numeric))
    if table == 'treatments':
        frame['invalid_interval'] = invalid_treatment_intervals(frame)
    return frame, report


def reconcile_same_day(frame):
    """Copy SDMT/functional visits; only unique non-null values survive per field.

    Full tables group by study_id/date; feature callers may supply an already
    patient-specific frame without study_id. Counts describe output group/field
    cells set missing, not the number of conflicting source rows. EDSS callers
    must retain their existing strict validation instead of using this helper.
    """
    counts = {'exact_duplicate_groups_collapsed': 0,
              'conflicting_patient_date_groups': 0,
              'conflicting_values_set_missing': 0}
    if frame.empty:
        return frame.copy(), counts
    keys = [c for c in ('study_id', 'date') if c in frame.columns]
    columns = [c for c in frame.columns if c not in keys]
    grouped = frame.groupby(keys, sort=False, dropna=False)
    distinct = grouped[columns].nunique(dropna=True)
    conflicts = distinct.gt(1)
    # first() is used only where there is at most ONE distinct non-null value.
    cleaned = grouped[columns].first().mask(conflicts).reset_index()
    unique_rows = frame.drop_duplicates().groupby(keys, sort=False, dropna=False).size()
    counts['exact_duplicate_groups_collapsed'] = int(((grouped.size() > 1) & unique_rows.eq(1)).sum())
    counts['conflicting_patient_date_groups'] = int(conflicts.any(axis=1).sum())
    counts['conflicting_values_set_missing'] = int(conflicts.to_numpy().sum())
    return cleaned, counts

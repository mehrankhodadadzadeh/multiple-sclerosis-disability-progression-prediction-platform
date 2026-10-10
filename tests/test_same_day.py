"""Same-day cleaning is per variable, historical only, and never applied to EDSS."""
import numpy as np
import pandas as pd
import pytest
from src.data.validation import reconcile_same_day
from src.features import cognition, functional, edss

T = pd.Timestamp('2020-01-01')


def test_exact_duplicates_nulls_and_patient_boundaries():
    frame = pd.DataFrame({'study_id': ['a', 'a', 'b', 'b', 'c', 'c'],
                          'date': [T] * 6, 'sdmt': [40., 40., 50., 60., np.nan, np.nan]})
    original = frame.copy(deep=True)
    cleaned, counts = reconcile_same_day(frame)
    assert len(cleaned) == 3
    assert cleaned.set_index('study_id').loc['a', 'sdmt'] == 40
    assert cleaned.set_index('study_id').loc[['b', 'c'], 'sdmt'].isna().all()
    assert counts == {'exact_duplicate_groups_collapsed': 2,
                      'conflicting_patient_date_groups': 1,
                      'conflicting_values_set_missing': 1}
    pd.testing.assert_frame_equal(frame, original)


def test_conflicting_sdmt_is_missing_in_features(config):
    frame = pd.DataFrame({'study_id': ['a', 'a'], 'date': [T, T], 'sdmt': [40., 50.]})
    result = cognition.extract(frame, T, config)
    assert np.isnan(result['latest_sdmt'])
    assert result['sdmt_count_24m'] == 0


def functional_conflict():
    frame = pd.DataFrame({'study_id': ['test-1'] * 2, 'date': [T, T],
                          't25fw_mean': [5., 5.], 'pasat3': [40., np.nan],
                          't25fw_trial1': [np.nan, np.nan], 't25fw_trial2': [np.nan, 5.]})
    for side in ('right', 'left'):
        for suffix in ('trial1', 'trial2', 'mean'):
            frame[f'nine_hpt_{side}_{suffix}'] = [20., 30.]
    return frame


def test_functional_conflicts_only_remove_affected_variables(config):
    frame = functional_conflict()
    original = frame.copy(deep=True)
    cleaned, counts = reconcile_same_day(frame)
    assert len(cleaned) == 1
    assert cleaned.filter(like='nine_hpt').isna().all().all()
    assert np.isnan(cleaned.iloc[0].t25fw_trial1)
    assert cleaned.iloc[0].t25fw_trial2 == 5
    assert cleaned.iloc[0].pasat3 == 40
    assert counts['conflicting_patient_date_groups'] == 1
    assert counts['conflicting_values_set_missing'] == 6
    result = functional.extract(frame, T, config)
    assert np.isnan(result['latest_9hpt_right']) and np.isnan(result['latest_9hpt_left'])
    assert result['latest_t25fw'] == 5 and result['latest_pasat'] == 40
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize('module', [cognition, functional])
def test_reconciliation_never_uses_future_visits(config, module):
    frame = (pd.DataFrame({'study_id': ['test-1'], 'date': [T], 'sdmt': [40.]})
             if module is cognition else functional_conflict().iloc[:1].copy())
    expected = module.extract(frame, T, config)
    future = pd.concat([frame, frame], ignore_index=True)
    future['date'] = T + pd.DateOffset(months=1)
    for col in future.select_dtypes(include='number'):
        future[col] = [1., 60.]
    combined = pd.concat([frame, future], ignore_index=True)
    pd.testing.assert_series_equal(pd.Series(module.extract(combined, T, config)), pd.Series(expected))


def test_edss_duplicates_and_conflict_rejection_unchanged(config):
    frame = pd.DataFrame({'study_id': ['a', 'a'], 'date': [T, T], 'edss': [2., 2.]})
    assert edss.extract(frame, T, config)['edss_count_24m'] == 1
    frame.loc[1, 'edss'] = 3.
    with pytest.raises(ValueError, match='Conflicting same-day'):
        edss.extract(frame, T, config)


def test_audit_counts_and_cohort_accept_handled_conflicts(tables, config, tmp_path):
    from src.data.quality import audit_tables, ensure_modeling_quality
    from src.cohort.builder import build_cohort
    tables['sdmt'] = pd.DataFrame({'study_id': ['test-1'] * 4,
        'date': pd.to_datetime(['2019-01-01'] * 2 + ['2020-01-01'] * 2),
        'sdmt': [30., 30., 40., 50.]})
    clinical = functional_conflict()
    exact = pd.concat([clinical.iloc[:1]] * 2, ignore_index=True)
    exact['date'] = pd.Timestamp('2019-01-01')
    tables['clinical_tests'] = pd.concat([exact, clinical], ignore_index=True).reindex(columns=tables['clinical_tests'].columns)
    originals = {name: tables[name].copy(deep=True) for name in ('sdmt', 'clinical_tests')}
    parsing = {name: {'dates': {}, 'invalid_numeric': {}} for name in tables}
    report = audit_tables(tables, parsing, config['data'], tmp_path)
    for name, count in [('sdmt', 1), ('clinical_tests', 6)]:
        assert report[name][f'{name}_exact_duplicate_groups_collapsed'] == 1
        assert report[name][f'{name}_conflicting_patient_date_groups'] == 1
        assert report[name][f'{name}_conflicting_values_set_missing'] == count
        assert report[name]['handled_quality_warnings']
    assert 'test-1' not in (tmp_path / 'data_quality.json').read_text()
    ensure_modeling_quality(tables, parsing, config['data'])
    cohort, _ = build_cohort(tables, config)
    assert not cohort.empty
    for name, original in originals.items():
        pd.testing.assert_frame_equal(tables[name], original)

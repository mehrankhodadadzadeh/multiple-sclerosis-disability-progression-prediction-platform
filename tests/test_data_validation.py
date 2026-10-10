import pandas as pd
import pytest
from src.data.validation import validate_columns,parse_dates,convert_table
from src.data.loaders import load_tables


def test_required_columns(config):
    with pytest.raises(ValueError,match='edss'):
        validate_columns(pd.DataFrame({'study_id':[],'date':[]}),config['data']['tables']['edss'],'edss')


def test_missing_files(tmp_path,config):
    with pytest.raises(ValueError,match='Missing required CSV'):
        load_tables(tmp_path,config['data'])


def test_dates_invalid_ambiguous_and_pivot():
    dates,report=parse_dates(pd.Series(['1/31/13','10/26/20','3/4/59','bad',None]))
    assert dates.iloc[0]==pd.Timestamp('2013-01-31')
    assert dates.iloc[2]==pd.Timestamp('1959-03-04')
    assert report=={'invalid':1,'ambiguous':1}
    assert dates.isna().sum()==2


def test_numeric_conversion_preserves_missing_and_reports_invalid(config):
    raw=pd.DataFrame({'study_id':['001','002','003'],'date':['2020-01-01']*3,'edss':['bad',None,'2.5']})
    frame,report=convert_table(raw,config['data']['tables']['edss'],config['data'],'edss')
    assert frame.edss.isna().sum()==2
    assert report['invalid_numeric']['edss']==1
    assert frame.study_id.iloc[0]=='001'


def test_invalid_config_rejected(config):
    from src.cohort.outcome import classify_outcome
    with pytest.raises(ValueError,match='thresholds'):
        classify_outcome(pd.DataFrame({'date':pd.to_datetime([]),'edss':[]}),pd.Timestamp('2020-01-01'),11,24,config['outcome'])


def test_aggregate_audit_contains_no_patient_identifiers(tables,config,tmp_path):
    from src.data.quality import audit_tables
    audit_tables(tables,{},config['data'],tmp_path)
    content=(tmp_path/'data_quality.json').read_text()
    assert 'test-1' not in content
    assert 'missingness_per_patient_distribution' in content
    assert 'categorical_fields_without_codebook' in content


def test_core_quality_errors_fail_closed(tables,config):
    from src.data.quality import ensure_modeling_quality
    parsing={name:{'dates':{},'invalid_numeric':{}} for name in tables}
    tables['edss'].loc[0,'edss']=11
    with pytest.raises(ValueError,match='outside configured range'):
        ensure_modeling_quality(tables,parsing,config['data'])


def test_handled_pasat_and_treatment_warnings_do_not_block_cohort(tables, config, tmp_path):
    from src.data.quality import audit_tables, ensure_modeling_quality
    from src.cohort.builder import build_cohort
    parsing = {name: {'dates': {}, 'invalid_numeric': {}} for name in tables}
    records = {
        'clinical_tests': pd.DataFrame({
            'study_id': ['test-1'] * 7,
            'date': pd.date_range('2018-01-01', periods=7).strftime('%Y-%m-%d'),
            'pasat3': ['0', '63', '78', '112', '-1', '1', '60']}),
        'treatments': pd.DataFrame({
            'study_id': ['test-1'] * 2,
            'start_date': ['2018-01-01', '2019-01-01'],
            'end_date': ['2017-12-01', None],
            'latest_date': ['2017-12-01', None],
            'status': ['Stopped', 'Active'], 'inn': ['invalid-drug', 'valid-drug']})}
    for name, raw in records.items():
        spec = config['data']['tables'][name]
        raw = raw.reindex(columns=spec['strings'] + spec['dates'] + spec['numbers']).astype('string')
        tables[name], parsing[name] = convert_table(raw, spec, config['data'], name)
    originals = {name: tables[name].copy(deep=True) for name in records}
    assert tables['treatments'].invalid_interval.tolist() == [True, False]
    assert tables['treatments'].loc[0, 'end_date'] == pd.Timestamp('2017-12-01')
    assert tables['treatments'].loc[0, 'latest_date'] == pd.Timestamp('2017-12-01')
    report = audit_tables(tables, parsing, config['data'], tmp_path)
    clinical = report['clinical_tests']
    assert clinical['pasat_zero_as_missing_count'] == 1
    assert clinical['pasat_above_60_count'] == 3
    assert clinical['pasat_negative_count'] == 1
    assert clinical['handled_quality_warnings']
    treatment_report = report['treatments']
    assert treatment_report['rows'] == 2
    assert treatment_report['invalid_treatment_intervals_count'] == 1
    assert treatment_report['excluded_treatment_intervals_count'] == 1
    assert treatment_report['handled_quality_warnings']
    ensure_modeling_quality(tables, parsing, config['data'])
    cohort, _ = build_cohort(tables, config)
    assert not cohort.empty
    assert 'invalid-drug' not in cohort.current_dmt.values
    for name, original in originals.items():
        pd.testing.assert_frame_equal(tables[name], original)


def test_overlap_audit_and_cohort_do_not_block(tables, config, tmp_path):
    from src.data.quality import audit_tables, ensure_modeling_quality
    from src.cohort.builder import build_cohort
    raw = pd.DataFrame({
        'study_id': ['test-1'] * 4,
        'start_date': ['2018-01-01', '2019-01-01', '2019-01-01', '2018-01-01'],
        'end_date': ['2020-01-01', '2021-01-01', None, '2017-01-01'],
        'inn': ['a', ' A ', 'b', 'invalid']})
    spec = config['data']['tables']['treatments']
    raw = raw.reindex(columns=spec['strings'] + spec['dates'] + spec['numbers']).astype('string')
    tables['treatments'], parsed = convert_table(raw, spec, config['data'], 'treatments')
    original = tables['treatments'].copy(deep=True)
    parsing = {name: {'dates': {}, 'invalid_numeric': {}} for name in tables}
    parsing['treatments'] = parsed
    report = audit_tables(tables, parsing, config['data'], tmp_path)['treatments']
    assert report['patients_with_overlapping_dmt_episodes'] == 1
    assert report['overlapping_dmt_episode_pairs'] == 3
    assert report['same_inn_overlap_pairs'] == 1
    assert report['different_inn_overlap_pairs'] == 2
    assert report['index_dates_with_multiple_active_dmts'] == 2
    assert report['invalid_treatment_intervals_count'] == 1
    assert len(report['handled_quality_warnings']) == 2
    ensure_modeling_quality(tables, parsing, config['data'])
    cohort, summary = build_cohort(tables, config)
    assert summary['index_dates_with_multiple_active_dmts'] == 2
    assert cohort.loc[cohort.index_date.eq(pd.Timestamp('2019-01-01')), 'active_dmt_count'].iloc[0] == 3
    assert cohort.loc[cohort.active_dmt_count.gt(1), 'current_dmt'].eq('MULTIPLE').all()
    assert cohort.loc[cohort.active_dmt_count.gt(1), 'current_dmt_duration_days'].isna().all()
    pd.testing.assert_frame_equal(tables['treatments'], original)
    assert 'test-1' not in (tmp_path / 'data_quality.json').read_text()

import numpy as np
import pandas as pd
import pytest
from src.features import edss,cognition,functional,treatment,relapse,demographics

T=pd.Timestamp('2020-01-01')


def test_edss_trajectory(tables,config):
    result=edss.extract(tables['edss'],T,config)
    assert result['current_edss']==2
    assert result['previous_edss']==1.5
    assert result['edss_change_12m']==0.5
    assert result['edss_change_24m']==1
    assert result['edss_count_24m']==3
    assert result['edss_slope_24m']==pytest.approx(0.5,abs=.002)


def test_missing_sdmt_stays_missing(tables,config):
    result=cognition.extract(tables['sdmt'],T,config)
    assert np.isnan(result['latest_sdmt'])
    assert result['sdmt_count_24m']==0


def test_change_no_future_reference_interpolation(config):
    frame=pd.DataFrame({'date':pd.to_datetime(['2019-02-01','2020-01-01']),'sdmt':[40,50]})
    assert np.isnan(cognition.extract(frame,T,config)['sdmt_change_12m'])


def test_functional_zero_policy(config):
    frame=pd.DataFrame({'date':[T],'t25fw_mean':[0],'nine_hpt_right_mean':[0],'nine_hpt_left_mean':[20],'pasat3':[0]})
    result=functional.extract(frame,T,config)
    assert np.isnan(result['latest_t25fw'])
    assert np.isnan(result['latest_9hpt_right'])
    assert result['latest_9hpt_left']==20
    assert np.isnan(result['latest_pasat'])


def test_treatment_history(config):
    frame=pd.DataFrame({'start_date':pd.to_datetime(['2018-01-01','2019-01-01','2021-01-01']),
                        'end_date':pd.to_datetime(['2018-12-31','2020-06-01',None]),'inn':['A','B','C']})
    result=treatment.extract(frame,T,config)
    assert result['current_dmt']=='B'
    assert result['number_previous_dmts']==1
    assert result['distinct_previous_dmts']==1
    assert result['dmt_starts_previous_24m']==2
    assert result['days_since_last_dmt_start']==365
    assert result['current_dmt_duration_days']==365


def test_overlapping_dmts_are_multiple(config):
    frame=pd.DataFrame({'start_date':pd.to_datetime(['2018-01-01','2019-01-01']),'end_date':pd.to_datetime([None,None]),'inn':['A','B']})
    result = treatment.extract(frame,T,config)
    assert result['currently_on_dmt'] == 1
    assert result['active_dmt_count'] == 2
    assert result['current_dmt'] == 'MULTIPLE'
    assert np.isnan(result['current_dmt_duration_days'])


def test_relapse_counts_and_unknown_corticosteroid(config):
    frame=pd.DataFrame({'start_date':pd.to_datetime(['2019-12-01','2021-01-01']),'corticosteroid':[1,1]})
    assert relapse.extract(frame,T,config)['relapse_therapy_count_12m']==1
    frame.loc[0,'corticosteroid']=np.nan
    assert np.isnan(relapse.extract(frame,T,config)['corticosteroid_episode_count_12m'])


def test_undated_smoking_and_future_onset_excluded(tables,config):
    patient=tables['patients'].iloc[0].copy();patient.ms_onset_date=pd.Timestamp('2021-01-01')
    result=demographics.extract(patient,T,config)
    assert np.isnan(result['disease_duration_years']) and np.isnan(result['smoking_status'])


@pytest.mark.parametrize('score', [0, -1, 63, 78, 112, 1, 60])
def test_pasat_cleaning_preserves_original_values(config, score):
    frame = pd.DataFrame({'date': [T], 't25fw_mean': [5.],
                          'nine_hpt_right_mean': [20.], 'nine_hpt_left_mean': [20.],
                          'pasat3': [score]})
    original = frame.copy(deep=True)
    result = functional.extract(frame, T, config)
    if 1 <= score <= 60:
        assert result['latest_pasat'] == score
    else:
        assert np.isnan(result['latest_pasat'])
    pd.testing.assert_frame_equal(frame, original)


def test_invalid_treatment_intervals_excluded_without_losing_valid_episodes(config):
    frame = pd.DataFrame({
        'start_date': pd.to_datetime(['2016-01-01', '2018-01-01', '2019-01-01']),
        'end_date': pd.to_datetime(['2017-01-01', '2017-12-01', None]),
        'inn': ['A', 'invalid-drug', 'B']})
    original = frame.copy(deep=True)
    for date in pd.to_datetime(['2017-06-01', '2018-06-01', '2020-01-01']):
        result = treatment.extract(frame, date, config)
        expected = treatment.extract(frame.iloc[[0, 2]], date, config)
        pd.testing.assert_series_equal(pd.Series(result), pd.Series(expected))
        invalid_only = treatment.extract(frame.iloc[[1]], date, config)
        assert invalid_only['currently_on_dmt'] == 0
        assert invalid_only['current_dmt'] == 'NONE'
        assert np.isnan(invalid_only['current_dmt_duration_days'])
        assert invalid_only['active_dmt_count'] == 0
        assert invalid_only['dmt_starts_previous_24m'] == 0
        assert np.isnan(invalid_only['days_since_last_dmt_start'])
    assert treatment.extract(frame, T, config)['current_dmt'] == 'B'
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize('name, expected', [('  interferon beta 1a  ', 'INTERFERON BETA 1A'),
                                           ('', 'UNKNOWN'), (None, 'UNKNOWN')])
def test_single_active_dmt_normalization(config, name, expected):
    frame = pd.DataFrame({'start_date': [pd.Timestamp('2019-01-01')],
                          'end_date': [pd.NaT], 'inn': [name]})
    original = frame.copy(deep=True)
    result = treatment.extract(frame, T, config)
    assert result['currently_on_dmt'] == 1
    assert result['active_dmt_count'] == 1
    assert result['current_dmt'] == expected
    assert result['current_dmt_duration_days'] == 365
    pd.testing.assert_frame_equal(frame, original)


def test_no_active_dmt_and_repeated_previous_drug(config):
    frame = pd.DataFrame({'start_date': pd.to_datetime(['2016-01-01', '2017-01-01']),
                          'end_date': pd.to_datetime(['2018-01-01', '2018-06-01']),
                          'inn': [' a ', 'A']})
    result = treatment.extract(frame, T, config)
    assert result['currently_on_dmt'] == 0
    assert result['active_dmt_count'] == 0
    assert result['current_dmt'] == 'NONE'
    assert np.isnan(result['current_dmt_duration_days'])
    assert result['number_previous_dmts'] == 2
    assert result['distinct_previous_dmts'] == 1
    assert result['dmt_starts_previous_24m'] == 0


def test_simultaneous_same_inn_episodes_are_not_merged(config):
    frame = pd.DataFrame({'start_date': [T, T], 'end_date': [pd.NaT, pd.NaT], 'inn': ['a', ' A ']})
    result = treatment.extract(frame, T, config)
    assert result['active_dmt_count'] == 2
    assert result['current_dmt'] == 'MULTIPLE'
    assert np.isnan(result['current_dmt_duration_days'])
    assert result['dmt_starts_previous_24m'] == 2
    assert result['days_since_last_dmt_start'] == 0

import pandas as pd
import pytest
from src.cohort.outcome import classify_outcome


def outcome(config,dates,values,baseline=2):
    return classify_outcome(pd.DataFrame({'date':pd.to_datetime(dates),'edss':values}),pd.Timestamp('2020-01-01'),baseline,24,config['outcome'])


def test_confirmed_positive(config):
    assert outcome(config,['2020-06-01','2020-09-01'],[3,3])['target']==1


@pytest.mark.parametrize('baseline,rise',[(0,1.5),(0.5,1),(5.5,1),(6,0.5),(9.5,0.5)])
def test_baseline_thresholds(config,baseline,rise):
    assert outcome(config,['2020-06-01','2020-09-01'],[baseline+rise]*2,baseline)['target']==1


def test_one_high_measurement_is_censored(config):
    assert outcome(config,['2020-06-01'],[3])['target'] is None


def test_no_future_is_censored_not_negative(config):
    assert outcome(config,[],[])['target'] is None


def test_stable_but_short_followup_is_censored(config):
    assert outcome(config,['2020-06-01','2021-01-01'],[2,2])['target'] is None


def test_adequate_negative(config):
    assert outcome(config,['2021-01-01','2022-01-01'],[2,2])['target']==0


def test_distant_followup_does_not_prove_negative(config):
    assert outcome(config,['2022-01-01'],[2])['outcome_reason']=='sparse_followup'


def test_confirmation_after_horizon_does_not_count(config):
    result=outcome(config,['2021-01-01','2021-12-01','2022-03-01'],[2,3,3])
    assert result['target'] is None


def test_intervening_recovery_breaks_confirmation(config):
    result=outcome(config,['2020-06-01','2020-07-01','2020-09-01','2020-10-01','2021-06-01','2022-01-01'],[3,2,3,2,2,2])
    assert result['target']==0


def test_confirmation_too_early_is_unknown(config):
    assert outcome(config,['2020-06-01','2020-07-01'],[3,3])['target'] is None


def test_configurable_confirmation_window(config):
    config['outcome']['confirmation_min_months']=6
    assert outcome(config,['2020-06-01','2020-09-01'],[3,3])['target'] is None
    assert outcome(config,['2020-06-01','2020-12-01'],[3,3])['target']==1


def test_duplicate_conflicting_edss_rejected(config):
    with pytest.raises(ValueError,match='Conflicting'):
        outcome(config,['2020-06-01','2020-06-01'],[3,2])


def test_worsening_at_post_horizon_boundary_is_not_negative(config):
    result=outcome(config,['2021-01-01','2021-07-01','2022-02-01'],[2,2,3])
    assert result['target'] is None
    assert result['outcome_reason']=='boundary_worsening_timing_uncertain'

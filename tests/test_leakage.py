import numpy as np
import pandas as pd
import pytest
from src.features import edss,cognition,functional,treatment,relapse
from src.modeling.pipelines import split_by_patient,make_pipeline,feature_frame
from src.cohort.builder import build_cohort

T=pd.Timestamp('2020-01-01')


def test_all_future_measurements_cannot_enter_features(tables,config):
    for module,name in [(edss,'edss'),(cognition,'sdmt'),(functional,'clinical_tests'),(treatment,'treatments'),(relapse,'relapse_therapy')]:
        before=module.extract(tables[name],T,config)
        changed=tables[name].copy()
        row={c:np.nan for c in changed.columns}
        row['study_id']='test-1'
        for c in config['data']['tables'][name]['dates']: row[c]=pd.Timestamp('2030-01-01')
        for c in config['data']['tables'][name]['numbers']: row[c]=999
        changed=pd.concat([changed,pd.DataFrame([row])],ignore_index=True)
        after=module.extract(changed,T,config)
        pd.testing.assert_series_equal(pd.Series(before),pd.Series(after))


def test_future_treatment_end_reason_status_do_not_leak(config):
    frame=pd.DataFrame({'start_date':[pd.Timestamp('2019-01-01')],'end_date':[pd.Timestamp('2021-01-01')],
                        'inn':['A'],'status':['finished'],'reason_for_stopping':['progression']})
    original=treatment.extract(frame,T,config)
    frame['end_date']=pd.NaT; frame['status']='active';frame['reason_for_stopping']=None
    pd.testing.assert_series_equal(pd.Series(original),pd.Series(treatment.extract(frame,T,config)))


def test_patient_split_has_no_overlap(config):
    frame=pd.DataFrame({'study_id':np.repeat([f'test-{i}' for i in range(20)],2),'target':[0,1]*20})
    splits=split_by_patient(frame,config['cohort'])
    groups=[set(s.study_id) for s in splits.values()]
    assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
    assert set.union(*groups)==set(frame.study_id)
    assert sum(map(len,splits.values()))==len(frame)
    again=split_by_patient(frame,config['cohort'])
    for key in splits: pd.testing.assert_frame_equal(splits[key],again[key])


def test_imputation_fits_training_only(config):
    model=make_pipeline('logistic_regression',['current_edss'],config['cohort'])
    train=pd.DataFrame({'current_edss':[1.,3.,np.nan,2.]})
    model.fit(train,[0,1,0,1])
    imputer=model.named_steps['preprocess'].named_transformers_['numeric'].named_steps['impute']
    assert imputer.statistics_[0]==2
    model.predict_proba(pd.DataFrame({'current_edss':[999.,np.nan]}))
    assert imputer.statistics_[0]==2


def test_cohort_censoring_and_historical_features(tables,config):
    cohort,report=build_cohort(tables,config)
    assert cohort.iloc[-1].outcome_status=='censored'
    assert pd.isna(cohort.iloc[-1].target)
    assert report['censored_samples']>0
    snapshot=cohort[cohort.index_date==T].iloc[0]
    assert snapshot.current_edss==2 and snapshot.edss_count_24m==3
    assert set(cohort.dropna(subset=['target']).target.unique())<={0,1}


@pytest.mark.parametrize('kind',['logistic_regression','random_forest','xgboost'])
def test_model_pipeline_and_tree_local_explanation(config,kind):
    config['cohort']['training'].update(forest_trees=3,xgb_trees=3,n_jobs=1)
    model=make_pipeline(kind,['current_edss','sex','latest_sdmt'],config['cohort'])
    frame=feature_frame(pd.DataFrame({'current_edss':[1.,2.,3.,4.,1.,5.],'sex':['F','M',None,'F','M','F'],'latest_sdmt':[np.nan]*6}),['current_edss','sex','latest_sdmt'])
    model.fit(frame,[0,0,1,1,0,1]);prob=model.predict_proba(frame)[:,1]
    assert np.isfinite(prob).all()
    if kind!='logistic_regression':
        from src.evaluation.explainability import local_explanation
        result=local_explanation(model,frame.iloc[:1])
        assert result['features']


def test_relapse_future_fields_do_not_change_episode_deduplication(config):
    frame=pd.DataFrame({'start_date':[pd.Timestamp('2019-01-01')]*2,'type':['acute']*2,'inn':['steroid']*2,'corticosteroid':[1,1],
                        'end_date':pd.to_datetime(['2021-01-01','2022-01-01']),'status':['complete','active']})
    before=relapse.extract(frame,T,config)
    frame['end_date']=pd.NaT;frame['status']='unknown'
    after=relapse.extract(frame,T,config)
    pd.testing.assert_series_equal(pd.Series(before),pd.Series(after))
    assert before['relapse_therapy_count_24m']==1


def test_configuration_cannot_add_target_or_identifier_to_features(config):
    from src.modeling.pipelines import feature_groups
    config['features']['groups']['A'].append('target')
    with pytest.raises(ValueError,match='cannot be configured as features'):
        feature_groups(config)

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from api import main
from src.evaluation.metrics import metrics,patient_bootstrap
from src.modeling.pipelines import make_pipeline


def test_api_schema_missing_values_and_no_echo(monkeypatch,config):
    model=make_pipeline('logistic_regression',['current_edss'],config['cohort'])
    model.fit(pd.DataFrame({'current_edss':[1.,2.,3.,4.]}),[0,0,1,1])
    metadata={'feature_list':['current_edss'],'feature_schema':{'current_edss':'numeric'},'risk_category_thresholds':[.2,.5],
              'model_version':'test','selected_model_id':'logistic','best_tree_model_id':'forest'}
    monkeypatch.setattr(main,'bundle',lambda:{'pipeline':model,'metadata':metadata})
    client=TestClient(main.app)
    assert client.get('/health').json()['model_loaded']
    assert client.get('/model-info').status_code==200
    for value in [2,None]:
        result=client.post('/predict',json={'features':{'current_edss':value},'include_explanation':True})
        assert result.status_code==200
        assert 0<=result.json()['progression_probability']<=1
        assert result.json()['explanation'] is None
    assert client.post('/predict',json={'features':{}}).status_code==422
    response=client.post('/predict',json={'features':{'current_edss':{'private':'secret'}}})
    assert response.status_code==422 and 'secret' not in response.text
    assert client.post('/predict',json={'features':{'current_edss':True}}).status_code==422
    assert client.post('/predict',json={'features':{'current_edss':'private'}}).status_code==422


def test_cluster_bootstrap_reproducible_and_finite():
    ids=np.array(['a','a','b','b','c','c','d','d'])
    y=np.array([0,0,1,1,0,1,0,1]);p=np.array([.1,.2,.8,.9,.2,.7,.2,.6])
    first=patient_bootstrap(y,p,ids,10,42,.5)
    assert first==patient_bootstrap(y,p,ids,10,42,.5)
    assert first['brier']['valid_iterations']==10
    assert first['auroc']['lower']==1
    result=metrics(y,p,.5,ids)
    assert result['sensitivity']==1 and result['specificity']==1


def test_full_training_orchestration_in_memory(tmp_path,monkeypatch,config):
    """Exercise every model/group and evaluation; never write a mock patient database."""
    from src.modeling import train
    from src.modeling.pipelines import feature_groups,CATEGORICAL
    config['cohort']['training'].update(forest_trees=3,xgb_trees=3,n_jobs=1,bootstrap_iterations=2,shap_max_samples=4)
    # Both outcomes per tiny mock patient ensure meaningful labels in every partition.
    ids=np.repeat([f'mock-{i}' for i in range(12)],2)
    target=np.tile([0,1],12)
    frame=pd.DataFrame({'study_id':ids,'target':target,'index_date':pd.to_datetime(['2020-01-01','2021-01-01']*12)})
    for name in feature_groups(config)['D']:
        frame[name]=np.where(target,'F','M') if name in CATEGORICAL else target.astype(float)+1
    saved={}
    original_to_csv=pd.DataFrame.to_csv
    def aggregate_csv_only(self,path,*args,**kwargs):
        if 'study_id' in self.columns: return None
        return original_to_csv(self,path,*args,**kwargs)
    monkeypatch.setattr(pd.DataFrame,'to_csv',aggregate_csv_only)
    monkeypatch.setattr(train.joblib,'dump',lambda obj,path:saved.update({str(path):obj}))
    monkeypatch.setattr(train,'save_model',lambda model,meta,path:saved.update({'model':model,'metadata':meta}))
    metadata=train.train_models(frame,config,tmp_path/'models',tmp_path/'reports',tmp_path/'artifacts')
    assert metadata['performance_metrics']['test']['brier']>=0
    assert len(pd.read_csv(tmp_path/'reports'/'model_comparison.csv'))==13
    assert (tmp_path/'reports'/'shap'/'global_shap_summary.png').exists()


def test_persisted_metadata_undefined_metrics_are_json_safe(tmp_path):
    import json
    from src.modeling.persistence import save_model,load_model
    save_model(None,{'performance_metrics':{'ppv':np.nan}},tmp_path)
    metadata=load_model(tmp_path)['metadata']
    assert metadata['performance_metrics']['ppv'] is None
    json.dumps(metadata,allow_nan=False)

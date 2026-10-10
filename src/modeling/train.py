"""Fixed patient split; validation-only model selection; one held-out test evaluation."""
from datetime import datetime, timezone
from pathlib import Path
import importlib.metadata
import hashlib
import json
import joblib
import pandas as pd
from src.config import write_json
from src.modeling.pipelines import feature_groups,feature_frame,split_by_patient,make_pipeline,CATEGORICAL
from src.modeling.persistence import save_model
from src.evaluation.metrics import metrics,patient_bootstrap
from src.evaluation.calibration import plot_evaluation
from src.evaluation.explainability import global_explanation


def train_models(cohort,config,model_dir,report_dir,artifact_dir):
    model_dir,report_dir,artifact_dir = map(Path,(model_dir,report_dir,artifact_dir))
    for p in (model_dir,report_dir,artifact_dir): p.mkdir(parents=True,exist_ok=True)
    # Split all snapshots first; retain a single assignment even for censored rows.
    partitions = split_by_patient(cohort,config['cohort'])
    assigned = pd.concat([part[['study_id','index_date']].assign(split=name) for name,part in partitions.items()])
    assigned.to_csv(artifact_dir/'patient_splits.csv',index=False)
    partitions = {name:part[part.target.notna()].copy() for name,part in partitions.items()}
    for name,part in partitions.items():
        if part.target.nunique()!=2: raise ValueError(f'{name} must contain both outcomes; insufficient cohort for the fixed split. Do not search seeds for better results.')
    write_json(report_dir/'split_summary.json',{name:{'patients':int(part.study_id.nunique()),'snapshots':len(part),'prevalence':float(part.target.mean())} for name,part in partitions.items()})
    tr,va,te = (partitions[k] for k in ('train','validation','test'))
    settings = config['cohort']['training']; seed=config['cohort']['random_seed']; threshold=settings['decision_threshold']
    candidates = [('baseline','edss_baseline',['current_edss'])]
    candidates += [(g,m,fs) for g,fs in feature_groups(config).items() for m in ('logistic_regression','random_forest','xgboost')]
    fitted,validation = {},[]
    for group,kind,features in candidates:
        key=f'{group}_{kind}'; model=make_pipeline(kind,features,config['cohort'])
        model.fit(feature_frame(tr,features),tr.target.astype(int))
        probability=model.predict_proba(feature_frame(va,features))[:,1]
        score=metrics(va.target,probability,threshold,va.study_id)
        validation.append({'model_id':key,'feature_group':group,'model_type':kind,**score})
        fitted[key]={'pipeline':model,'features':features,'feature_group':group,'model_type':kind}
        print(f'Fitted {key} (aggregate validation Brier: {score["brier"]:.4f})')
    # Brier prioritizes probability quality; AUPRC breaks ties. Never inspect test to select.
    if settings['selection_metric'] != 'brier': raise ValueError('V1 supports prespecified Brier selection only')
    ranked=sorted(validation,key=lambda r:(r['brier'],-r['auprc']))
    best_id=ranked[0]['model_id']
    best_tree_id=next(r['model_id'] for r in ranked if r['model_type'] in ('random_forest','xgboost'))
    metadata={'model_type':fitted[best_id]['model_type'],'feature_group':fitted[best_id]['feature_group'],
              'feature_list':fitted[best_id]['features'],'training_date':datetime.now(timezone.utc).isoformat(),
              'outcome_configuration':config['outcome'],'prediction_horizon_months':config['cohort']['prediction_horizon_months'],
              'random_seed':seed,'decision_threshold':threshold,'risk_category_thresholds':settings['risk_category_thresholds'],
              'selected_model_id':best_id,'best_tree_model_id':best_tree_id,'selection_rule':'Lowest validation Brier; AUPRC tie-break',
              'feature_schema':{f:('categorical' if f in CATEGORICAL else 'numeric') for f in fitted[best_id]['features']},
              'configuration':config,'package_versions':{name:importlib.metadata.version(name) for name in ('numpy','pandas','scikit-learn','xgboost','shap','joblib')},
              'disclaimer':'Research prototype only. Not validated for clinical decision-making. Not intended to diagnose MS progression or recommend treatment.'}
    metadata['model_version']=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()[:12]
    # Private evaluation cache supports exact re-evaluation; never publish this artifact.
    evaluation={'models':fitted,'validation':validation,'test':te,'config':config,'metadata':metadata}
    joblib.dump(evaluation,artifact_dir/'evaluation_cache.joblib')
    results=evaluate_models(evaluation,report_dir)
    metadata['performance_metrics']={'validation':next(r for r in validation if r['model_id']==best_id),
                                     'test':next(r for r in results if r['model_id']==best_id)}
    tree=fitted[best_tree_id]
    global_explanation(tree['pipeline'],feature_frame(va,tree['features']),report_dir/'shap',settings['shap_max_samples'],seed)
    write_json(report_dir/'shap'/'explanation_metadata.json',{'model_id':best_tree_id,'feature_group':tree['feature_group'],'sample_source':'validation','is_inference_model':best_tree_id==best_id})
    joblib.dump(tree,model_dir/'best_tree_model.joblib')
    save_model(fitted[best_id]['pipeline'],metadata,model_dir)
    return metadata


def evaluate_models(cache,report_dir):
    config=cache['config']; settings=config['cohort']['training']; te=cache['test']; threshold=settings['decision_threshold']
    report_dir=Path(report_dir); results=[]; intervals={}
    for key,item in cache['models'].items():
        p=item['pipeline'].predict_proba(feature_frame(te,item['features']))[:,1]
        score=metrics(te.target,p,threshold,te.study_id)
        results.append({'model_id':key,'feature_group':item['feature_group'],'model_type':item['model_type'],**score})
        intervals[key]=patient_bootstrap(te.target,p,te.study_id,settings['bootstrap_iterations'],config['cohort']['random_seed'],threshold,settings['confidence_level'])
        plot_evaluation(te.target,p,threshold,report_dir/'evaluation'/key)
    pd.DataFrame(cache['validation']).assign(split='validation').to_csv(report_dir/'validation_comparison.csv',index=False)
    pd.DataFrame(results).assign(split='test').to_csv(report_dir/'model_comparison.csv',index=False)
    write_json(report_dir/'confidence_intervals.json',intervals)
    write_json(report_dir/'evaluation_summary.json',{'selected_model_id':cache['metadata']['selected_model_id'],'test_results':results,'bootstrap_iterations':settings['bootstrap_iterations'],'confidence_level':settings['confidence_level']})
    return results

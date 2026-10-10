"""Local-only inference for one already-engineered record. Never log request bodies."""
import math
import os
from functools import lru_cache
import pandas as pd
from fastapi import FastAPI,HTTPException,Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel,ConfigDict
from src.modeling.persistence import load_model
from src.modeling.pipelines import feature_frame

app=FastAPI(title='Multiple Sclerosis Disability Progression Prediction Platform',version='1.0')


class PredictionRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    features: dict[str,float|str|None]
    include_explanation: bool=False


@app.exception_handler(RequestValidationError)
async def invalid_request(request:Request,exc:RequestValidationError):
    # Default FastAPI validation errors can echo confidential submitted values.
    return JSONResponse(status_code=422,content={'detail':'Invalid prediction request; use the exact feature schema from /model-info.'})


@lru_cache(maxsize=1)
def bundle():
    try: return load_model(os.getenv('MS_MODEL_DIR','models'))
    except FileNotFoundError: raise HTTPException(503,'No trained model available. Complete the local training workflow.') from None


@app.get('/health')
def health():
    try: bundle(); ready=True
    except HTTPException: ready=False
    return {'status':'ok','model_loaded':ready}


@app.get('/model-info')
def model_info(): return bundle()['metadata']


@app.post('/predict')
def predict(payload:PredictionRequest):
    stored=bundle(); meta=stored['metadata']; features=meta['feature_list']
    if set(payload.features)!=set(features): raise HTTPException(422,'Feature keys must exactly match /model-info; use null for missing values.')
    for name,kind in meta['feature_schema'].items():
        value=payload.features[name]
        if value is None: continue
        if kind=='numeric' and (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)):
            raise HTTPException(422,'Numeric features must be finite numbers or null.')
        if kind=='categorical' and not isinstance(value,str): raise HTTPException(422,'Categorical features must be strings or null.')
    frame=feature_frame(pd.DataFrame([payload.features]),features)
    probability=float(stored['pipeline'].predict_proba(frame)[0,1])
    low,high=meta['risk_category_thresholds']
    result={'progression_probability':probability,'risk_category':'low' if probability<low else 'intermediate' if probability<high else 'high','model_version':meta['model_version']}
    if payload.include_explanation:
        if meta['selected_model_id']==meta['best_tree_model_id']:
            from src.evaluation.explainability import local_explanation
            result['explanation']=local_explanation(stored['pipeline'],frame)
        else:
            result['explanation']=None
            result['explanation_note']='SHAP is available for the best tree model only; the selected inference model differs.'
    return result

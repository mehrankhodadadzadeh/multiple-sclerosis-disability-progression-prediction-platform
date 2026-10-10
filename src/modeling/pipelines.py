"""All learned preprocessing lives inside training-only sklearn pipelines."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

CATEGORICAL = {'sex','current_dmt','smoking_status'}


def feature_groups(config):
    groups, accumulated = {}, []
    for name in ('A','B','C','D'):
        requested = config['features']['groups'][name]
        forbidden = {'study_id','index_date','target','outcome_status','outcome_reason'}
        if forbidden.intersection(requested):
            raise ValueError('Identifiers, outcome labels and follow-up metadata cannot be configured as features')
        accumulated += requested
        groups[name] = list(dict.fromkeys(accumulated))
    return groups


def feature_frame(frame, features):
    result = frame[features].copy()
    for col in features:
        if col in CATEGORICAL:
            result[col] = result[col].astype(object).where(result[col].notna(),np.nan)
        else: result[col] = pd.to_numeric(result[col],errors='raise').astype(float)
    return result


def split_by_patient(frame, config):
    fractions = config['split']; seed = config['random_seed']
    train, rest = next(GroupShuffleSplit(n_splits=1,train_size=fractions['train'],random_state=seed).split(frame,groups=frame.study_id))
    other = frame.iloc[rest]
    validation, test = next(GroupShuffleSplit(n_splits=1,train_size=fractions['validation']/(fractions['validation']+fractions['test']),random_state=seed+1).split(other,groups=other.study_id))
    partitions = {'train':frame.iloc[train].copy(), 'validation':other.iloc[validation].copy(), 'test':other.iloc[test].copy()}
    ids = [set(x.study_id) for x in partitions.values()]
    if any(ids[i] & ids[j] for i in range(3) for j in range(i+1,3)): raise AssertionError('Patient overlap')
    return partitions


def make_pipeline(model_type,features,config):
    numeric = [x for x in features if x not in CATEGORICAL]
    categorical = [x for x in features if x in CATEGORICAL]
    numeric_pipe = Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())])
    categorical_pipe = Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='Missing/Unknown',keep_empty_features=True)),
                                 ('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])
    preprocess = ColumnTransformer([('numeric',numeric_pipe,numeric),('categorical',categorical_pipe,categorical)],verbose_feature_names_out=False)
    p = config['training']; seed = config['random_seed']
    if model_type in ('logistic_regression','edss_baseline'):
        model = LogisticRegression(max_iter=p['logistic_max_iter'],random_state=seed)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=p['forest_trees'],min_samples_leaf=p['forest_min_samples_leaf'],random_state=seed,n_jobs=p['n_jobs'])
    elif model_type == 'xgboost':
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=p['xgb_trees'],max_depth=p['xgb_max_depth'],learning_rate=p['xgb_learning_rate'],
                              objective='binary:logistic',eval_metric='logloss',random_state=seed,n_jobs=p['n_jobs'],tree_method='hist')
    else: raise ValueError('Unsupported model type')
    return Pipeline([('preprocess',preprocess),('model',model)])

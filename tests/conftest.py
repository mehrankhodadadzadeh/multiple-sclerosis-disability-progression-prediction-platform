from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from src.config import load_config


@pytest.fixture
def config(): return load_config(Path(__file__).resolve().parents[1]/'configs')


@pytest.fixture
def tables(config):
    result={}
    for name,spec in config['data']['tables'].items():
        result[name]=pd.DataFrame({**{c:pd.Series(dtype='object') for c in spec['strings']},
                                  **{c:pd.Series(dtype='datetime64[ns]') for c in spec['dates']},
                                  **{c:pd.Series(dtype='float64') for c in spec['numbers']}})
    p={c:np.nan for c in result['patients'].columns}
    p.update(study_id='test-1',date_of_birth=pd.Timestamp('1980-01-01'),ms_onset_date=pd.Timestamp('2005-01-01'),sex='F',smoking_status='yes')
    result['patients']=pd.DataFrame([p])
    for c in config['data']['tables']['patients']['dates']: result['patients'][c]=pd.to_datetime(result['patients'][c])
    result['edss']=pd.DataFrame({'study_id':['test-1']*5,'date':pd.to_datetime(['2018-01-01','2019-01-01','2020-01-01','2021-01-01','2022-01-01']),'edss':[1.,1.5,2.,2.,2.]})
    return result

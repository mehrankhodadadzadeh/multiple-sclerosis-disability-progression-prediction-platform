"""Manual research demo; only communicates with a loopback FastAPI service."""
import json
import os
from pathlib import Path
from urllib.parse import urlparse
os.environ.setdefault('STREAMLIT_BROWSER_GATHER_USAGE_STATS','false')
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title='MS Disability Progression',layout='wide')
st.title('Multiple Sclerosis Disability Progression Prediction Platform')
st.warning('Research prototype only. Not validated for clinical decision-making. Not intended to diagnose MS progression or recommend treatment.')
page=st.sidebar.radio('Section',['Project overview','Dataset/cohort summary','Model performance','Single-patient prediction demo'])
reports=Path(os.getenv('MS_REPORT_DIR','reports'))
api=os.getenv('MS_API_URL','http://127.0.0.1:8000').rstrip('/')
parsed=urlparse(api)
if parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','localhost','::1') or parsed.username or parsed.password:
    st.error('The API address must be a local loopback HTTP address.'); st.stop()
session=requests.Session(); session.trust_env=False  # Do not route health data through proxy environment variables.

if page=='Project overview':
    st.write('Research question: can routinely collected longitudinal clinical data predict confirmed EDSS disability progression during the following 24 months?')
    st.write('Model A uses age, sex, disease duration and current EDSS. B adds EDSS history; C adds treatment and relapse-treatment history; D adds cognitive and functional tests.')
    st.write('This outcome is not PIRA. All snapshots from one patient stay in one split. Unknown outcomes are excluded from binary training.')
elif page=='Dataset/cohort summary':
    for name in ['cohort_summary.json','data_quality.json']:
        path=reports/name
        if path.exists():
            with st.expander(name,expanded=name=='cohort_summary.json'): st.json(json.loads(path.read_text()))
        else: st.info('Run the local audit and cohort build to populate aggregate reports.')
elif page=='Model performance':
    path=reports/'model_comparison.csv'
    if path.exists():
        table=pd.read_csv(path); st.dataframe(table,hide_index=True)
        selected=st.selectbox('View held-out plots',table.model_id.tolist())
        for name in ['roc','precision_recall','calibration','confusion_matrix']:
            image=reports/'evaluation'/selected/f'{name}.png'
            if image.exists(): st.image(str(image))
        intervals=reports/'confidence_intervals.json'
        if intervals.exists(): st.json(json.loads(intervals.read_text()).get(selected,{}))
    else: st.info('Train and evaluate locally to view performance.')
else:
    st.caption('Enter one already-engineered historical snapshot. Do not enter names, identifiers or patient histories. Blank fields represent missing values.')
    try:
        response=session.get(api+'/model-info',timeout=10); response.raise_for_status(); info=response.json()
    except (requests.RequestException,ValueError):
        st.info('Start the local API with a trained model to enable this demo.'); st.stop()
    horizon=info['prediction_horizon_months']
    st.write(f"Model: {info['model_type']} · feature group {info['feature_group']} · horizon {horizon} months")
    st.caption('Risk categories use provisional research thresholds, not validated clinical action cutoffs.')
    with st.form('prediction'):
        entered={name:st.text_input(name.replace('_',' '),key=name) for name in info['feature_list']}
        submit=st.form_submit_button('Estimate progression risk')
    if submit:
        try:
            values={k:(None if not v.strip() else float(v) if info['feature_schema'][k]=='numeric' else v.strip()) for k,v in entered.items()}
            response=session.post(api+'/predict',json={'features':values,'include_explanation':True},timeout=30)
            response.raise_for_status(); result=response.json()
            st.metric(f'Estimated {horizon}-month progression risk',f"{result['progression_probability']:.1%}")
            st.write('Risk category: '+result['risk_category'])
            if result.get('explanation'):
                st.subheader('Features contributing to the estimated risk')
                st.dataframe(pd.DataFrame(result['explanation']['features']),hide_index=True)
                st.caption(result['explanation']['note'])
            elif result.get('explanation_note'): st.caption(result['explanation_note'])
        except (ValueError,requests.RequestException): st.error('Prediction failed. Check numeric inputs and local API availability.')

"""SHAP for the validation-selected best tree only. Associations, never causes."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def shap_values(pipeline,frame):
    import shap
    transformed = pipeline.named_steps['preprocess'].transform(frame)
    names = pipeline.named_steps['preprocess'].get_feature_names_out()
    explainer = shap.TreeExplainer(pipeline.named_steps['model'])
    values = explainer.shap_values(transformed,check_additivity=False)
    if isinstance(values,list): values = values[1]
    values = np.asarray(values)
    if values.ndim==3: values = values[:,:,1]
    return transformed,names,values


def global_explanation(pipeline,frame,directory,max_samples,seed):
    import shap
    directory = Path(directory); directory.mkdir(parents=True,exist_ok=True)
    sample = frame.sample(min(len(frame),max_samples),random_state=seed)
    x,names,values = shap_values(pipeline,sample)
    importance = pd.DataFrame({'feature':names,'mean_absolute_shap':np.abs(values).mean(axis=0)}).sort_values('mean_absolute_shap',ascending=False)
    importance.to_csv(directory/'top_feature_importance.csv',index=False)
    shap.summary_plot(values,x,feature_names=names,show=False,max_display=20)
    plt.tight_layout(); plt.savefig(directory/'global_shap_summary.png',bbox_inches='tight'); plt.close('all')
    for name in importance.feature.head(2):
        shap.dependence_plot(list(names).index(name),values,x,feature_names=names,interaction_index=None,show=False)
        plt.savefig(directory/f'dependence_{list(names).index(name)}.png',bbox_inches='tight'); plt.close('all')
    return importance


def local_explanation(pipeline,frame,limit=8):
    _,names,values = shap_values(pipeline,frame)
    order = np.argsort(np.abs(values[0]))[::-1][:limit]
    return {'title':'Features contributing to the estimated risk',
            'note':'Associations within this model, not causal effects. SHAP units depend on model: probability for random forest; log-odds for XGBoost.',
            'features':[{'feature':str(names[i]),'contribution':float(values[0,i]),'direction':'higher' if values[0,i]>0 else 'lower' if values[0,i]<0 else 'neutral'} for i in order]}

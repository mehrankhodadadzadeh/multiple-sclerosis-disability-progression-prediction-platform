import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, f1_score, brier_score_loss


def metrics(y,probability,threshold=0.5,patient_ids=None):
    y, p = np.asarray(y,dtype=int), np.asarray(probability)
    prediction = p >= threshold
    tn,fp,fn,tp = confusion_matrix(y,prediction,labels=[0,1]).ravel()
    def ratio(a,b): return float(a/b) if b else np.nan
    result = {'auroc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan,
              'auprc':float(average_precision_score(y,p)) if len(np.unique(y))==2 else np.nan,
              'sensitivity':ratio(tp,tp+fn),'specificity':ratio(tn,tn+fp),'ppv':ratio(tp,tp+fp),'npv':ratio(tn,tn+fn),
              'f1':float(f1_score(y,prediction,zero_division=0)), 'brier':float(brier_score_loss(y,p)),
              'snapshot_alert_fraction':float(prediction.mean()),'alerts_per_true_progression':ratio(tp+fp,tp),
              'prevalence':float(y.mean())}
    if patient_ids is not None:
        ids = np.asarray(patient_ids)
        result['patient_alert_fraction'] = float(np.unique(ids[prediction]).size / np.unique(ids).size)
    return result


def patient_bootstrap(y,p,patient_ids,iterations,seed,threshold,confidence_level=0.95):
    y,p,ids = np.asarray(y),np.asarray(p),np.asarray(patient_ids)
    unique = np.unique(ids); rng = np.random.default_rng(seed)
    positions = {sid:np.flatnonzero(ids==sid) for sid in unique}
    samples = []
    for _ in range(iterations):
        selected = rng.choice(unique,size=len(unique),replace=True)
        index = np.concatenate([positions[sid] for sid in selected])
        # Repeated clusters receive distinct bootstrap patient identities.
        cluster_ids = np.concatenate([np.full(len(positions[sid]),i) for i,sid in enumerate(selected)])
        samples.append(metrics(y[index],p[index],threshold,cluster_ids))
    output = {}
    for key in metrics(y,p,threshold,ids):
        values = np.array([row[key] for row in samples]); values = values[np.isfinite(values)]
        alpha = (1-confidence_level)/2
        output[key] = {'lower':float(np.quantile(values,alpha)) if len(values) else None,
                       'upper':float(np.quantile(values,1-alpha)) if len(values) else None,
                       'valid_iterations':len(values)}
    return output

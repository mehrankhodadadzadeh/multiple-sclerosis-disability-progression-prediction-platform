"""Held-out plots; calibration is assessed without fitting on held-out labels."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
from sklearn.calibration import CalibrationDisplay


def plot_evaluation(y,p,threshold,directory):
    directory = Path(directory); directory.mkdir(parents=True,exist_ok=True)
    for name, display in [('roc',RocCurveDisplay),('precision_recall',PrecisionRecallDisplay),('calibration',CalibrationDisplay),('confusion_matrix',ConfusionMatrixDisplay)]:
        if name in ('roc','precision_recall') and len(np.unique(y)) < 2: continue
        fig,ax = plt.subplots(figsize=(5,4))
        if name=='confusion_matrix': display.from_predictions(y,np.asarray(p)>=threshold,labels=[0,1],ax=ax,colorbar=False)
        elif name=='calibration': display.from_predictions(y,p,n_bins=10,strategy='quantile',ax=ax)
        else: display.from_predictions(y,p,ax=ax)
        fig.tight_layout(); fig.savefig(directory/f'{name}.png'); plt.close(fig)

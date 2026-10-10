import joblib
from common import arguments,load_config,fingerprint,run
from src.modeling.train import train_models


def main():
    parser=arguments('Train only after reviewing the aggregate cohort report',data=True)
    parser.add_argument('--cohort-reviewed',action='store_true',help='Attest that the cohort, censoring, quality report and clinical configuration have been reviewed')
    args=parser.parse_args(); config=load_config(args.config_dir)
    if not args.cohort_reviewed:
        raise ValueError('Training requires --cohort-reviewed after reviewing the cohort report (DATA_INFORMATION.md section 25).')
    saved=joblib.load(args.artifact_dir/'cohort.joblib')
    if saved['fingerprint']!=fingerprint(args.data_dir,config):
        raise ValueError('Input data or configuration changed. Rebuild and review the cohort before training.')
    metadata=train_models(saved['cohort'],config,args.model_dir,args.report_dir,args.artifact_dir)
    print(f'Saved validation-selected model: {metadata["selected_model_id"]}')

if __name__=='__main__': run(main)

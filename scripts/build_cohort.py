import joblib
from common import arguments,load_config,fingerprint,run
from src.data.loaders import load_tables
from src.data.quality import audit_tables,ensure_modeling_quality
from src.cohort.builder import build_cohort


def main():
    args=arguments('Build historical snapshots and censoring report; stop for review',data=True).parse_args()
    config=load_config(args.config_dir)
    tables,parsing=load_tables(args.data_dir,config['data'])
    audit_tables(tables,parsing,config['data'],args.report_dir)
    ensure_modeling_quality(tables,parsing,config['data'])
    cohort,summary=build_cohort(tables,config,args.report_dir)
    args.artifact_dir.mkdir(parents=True,exist_ok=True)
    joblib.dump({'cohort':cohort,'fingerprint':fingerprint(args.data_dir,config),'configuration':config},args.artifact_dir/'cohort.joblib')
    print(f'Built {len(cohort)} snapshots; {summary["censored_samples"]} censored. Review reports/cohort_summary.json in your report directory before training.')

if __name__=='__main__': run(main)

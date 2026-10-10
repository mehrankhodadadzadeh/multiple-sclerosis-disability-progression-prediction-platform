from common import arguments,load_config,run
from src.data.loaders import load_tables
from src.data.quality import audit_tables


def main():
    args=arguments('Audit nine local processed CSV tables',data=True).parse_args()
    config=load_config(args.config_dir)
    tables,parsing=load_tables(args.data_dir,config['data'])
    audit_tables(tables,parsing,config['data'],args.report_dir)
    print('Aggregate audit written. Review date assumptions, missingness and quality flags before cohort construction.')

if __name__=='__main__': run(main)

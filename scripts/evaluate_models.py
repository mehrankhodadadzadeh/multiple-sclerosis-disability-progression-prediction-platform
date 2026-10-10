import joblib
from common import arguments,run
from src.modeling.train import evaluate_models


def main():
    args=arguments('Reproduce held-out evaluation using the trusted local evaluation cache').parse_args()
    evaluate_models(joblib.load(args.artifact_dir/'evaluation_cache.joblib'),args.report_dir)
    print('Aggregate held-out metrics, cluster bootstrap intervals and plots written.')

if __name__=='__main__': run(main)

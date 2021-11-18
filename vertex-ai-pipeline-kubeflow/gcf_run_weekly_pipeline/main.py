from kfp.v2.google.client import AIPlatformClient
from datetime import datetime

project_id = 'project-id'
region = 'us-central1'
pipeline_name = 'earthquake-prediction'

main_bucket_name = 'bucket-name'
pipeline_root_path = 'gs://' + main_bucket_name + '/pipelines/'

model_name = '5390f2dc404cbe0cd01427a938160354'
problem_statement_file = 'problem_statement/weekly.json'
raw_location = 'data/quakes/raw/'
silver_location = 'data/quakes/silver/'
gold_location = 'data/quakes/gold/' + model_name + '/'
artifact_location = 'models/' + model_name + '/'
predictions_location = 'data/predictions/' + model_name + '/'
metrics_location = 'data/metrics/' + model_name + '/'

def run_pipeline(request):

    api_client = AIPlatformClient(
        project_id = project_id,
        region = region
    )

    for i in range(7):

        parameter_values = {
            'is_training': 'False',
            'maint_bucket': main_bucket_name,
            'problem_statement_file': problem_statement_file,
            'date_offset': i,
            'raw_location': raw_location,
            'silver_location': silver_location,
            'gold_location': gold_location,
            'artifact_location': artifact_location,
            'predictions_location': predictions_location,
            'metrics_location': metrics_location,
        }

        run_time = datetime.now().strftime('%Y%m%d%H%m%S%f')

        api_client.create_run_from_job_spec(
            job_spec_path = pipeline_root_path + 'json/' + pipeline_name.replace('-', '_') + '.json',
            job_id = pipeline_name.replace('-', '') + '{0}'.format(run_time),
            pipeline_root = pipeline_root_path,
            enable_caching = False,
            parameter_values = parameter_values
        )

    return 'Pipeline run successfully'
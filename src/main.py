import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv

from utils import read_config, yolov8nms
from train import Trainer

from ultralytics import settings
import shutil



load_dotenv()
MLFLOW_TRACKING_URI=os.getenv('MLFLOW_TRACKING_URI')


if __name__ == '__main__':

    try:
        settings.update({"mlflow": False})
    except Exception as e:
        print(f"Error updating settings: {e}")

    params = read_config('params.yaml')

    # set the tracking uri 
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(params['experiment']['experiment_name'])

    # start mlflow experiment 
    with mlflow.start_run(run_name=params['experiment']['name']) as run:
        
        results = Trainer(params).train()

        if params['format'] == 'torchscript':
            path = results.save_dir/'weights'
            os.environ['WEIGHTS_PATH'] = str(path)
            modelnms=yolov8nms(str(path/'best.torchscript'))
            modelnms.save(str(path/'best.nms_torchscript'))
        
        ##TODO: register the model with mlflow

        
        mlflow.log_artifacts(str(results.save_dir))

        # log params with mlflow
        mlflow.log_param('model_type', params['experiment']['model_type'])
        mlflow.log_param('epochs',params['parameters']['epochs'])
        mlflow.log_param('optimizer', params['parameters']['optimizer'])
        mlflow.log_param('learning_rate', params['parameters']['lr0'])

        # log metrics with mlflow
        mlflow.log_metric("metric/recall",results.results_dict['metrics/recall(B)'])
        mlflow.log_metric("metric/precision",results.results_dict['metrics/precision(B)'])
        mlflow.log_metric("metric/mAP_0.5",results.results_dict['metrics/mAP50(B)'])
        mlflow.log_metric("loss",results.speed['loss'])

        mlflow.log_metric("speed/preprocess",results.speed['preprocess'])
        mlflow.log_metric("speed/inference",results.speed['inference'])
        mlflow.log_metric("speed/postprocess",results.speed['postprocess'])

        mlflow.log_artifact(params['data'])
        mlflow.log_artifact("params.yaml")

        runs_dir = Path('runs')
        if runs_dir.exists() and runs_dir.is_dir():
            shutil.rmtree(runs_dir)



         











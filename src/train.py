
from ultralytics import YOLO
import torch
class Trainer:
    def __init__(self, config):
        self.params = config
        self.DEVICE = self.params['parameters']['device'] if self.params['parameters']['device']\
        else ('cuda:0' if torch.cuda.is_available() else 'cpu')

    def train(self):
        params = self.params

        if params['weights']:
            model = YOLO(params['weights'])
        else:
            model = YOLO(params['experiment']['model_type'])

        if params['resume']:
            results=model.train(resume=True)
        else:
            results = model.train(
                data=params['data'],
                imgsz=params['parameters']['imgsz'],
                batch=params['parameters']['batch'],
                epochs=params['parameters']['epochs'],
                optimizer=params['parameters']['optimizer'],
                lr0=params['parameters']['lr0'],
                seed=params['parameters']['seed'],
                pretrained=params['parameters']['pretrained'],
                device = self.DEVICE,
                save = params['parameters']['save']
        )

        if params['format'] == 'torchscript':
            model.export(format='torchscript')
        
        return results


from ultralytics import YOLO
class Trainer:
    def __init__(self, config):
        self.param = config
        self.DEVICE = self.params['device'] if self.params['device'] else 'cpu'

    def train(self, epoch):
        params = self.param

        if params['weights']:
            model = YOLO(params['weights'])
        else:
            model = YOLO(params['model_type'])

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
                name=params['parameters']['name'],
                device = self.DEVICE,
                save = params['parameters']['save']
        )

        if params['format'] == 'torchscript':
            model.export(format='torchscript')
        
        return results

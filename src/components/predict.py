import argparse
import torch
from torchvision.io import read_image
import torchvision.transforms as T
import cv2
from ultralytics import YOLO

if __name__ == '__main__':
    argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Predict using YOLOv8 model')
    parser.add_argument('--model_path', type=str, help='Path to the model')
    parser.add_argument('--image_path', type=str, help='Path to the image')
    args = parser.parse_args()

    model_path = args.model_path
    image_path = args.image_path

    if model_path.endswith('.nms_torchscript'):

        model = torch.jit.load(model_path)
        model.eval()
        image = read_image(image_path)
        orig_size = image.shape[1:]
        mask_size = (640,640)
        image = T.Resize(mask_size,antialias=False)(image)
        image = image.float()
        image = image/255
        # image = image.permute(2,0,1)
        image = image.unsqueeze(0)
        conf_iou = torch.tensor(0.5,dtype = torch.float)
        conf = torch.tensor(0.1, dtype=torch.float)
        alpha = 0.6
        cutout = 20
        preds = model(image,conf,conf_iou)

        boxes,masks,pixels,img,crop = preds
        ##TODO: Display the results for torchscript model
        # out = img.squeeze(0).permute(1,2,0).numpy()
        # out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        # cv2.imshow('image',out)
        # cv2.waitKey(0)
    else:
        model = YOLO(model_path)
        results = model(image_path)
        ##TODO: Display the results for yolo model


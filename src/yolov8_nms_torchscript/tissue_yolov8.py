import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.mobile_optimizer import optimize_for_mobile
from typing import List
import numpy as np
import cv2
from torchvision.io import read_image
import torchvision.transforms as T


class Yolov8NMS(nn.Module):
    def __init__(self, model:torch.jit._script.RecursiveScriptModule):
        super(Yolov8NMS, self).__init__()
        self.model = model
        self.names = {}        
    
    def preprocess(self, input_tensor: torch.FloatTensor):

        image = input_tensor.float()
        image = image/255
        # image = image.permute(2,0,1)
        image = image.unsqueeze(0)

        return image
    
    def xywh2xyxy(self,x):
        assert x.shape[-1] == 4, f'input shape last dimension expected 4 but input shape is {x.shape}'
        y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
        dw = x[..., 2] / 2  # half-width
        dh = x[..., 3] / 2  # half-height
        y[..., 0] = x[..., 0] - dw  # top left x
        y[..., 1] = x[..., 1] - dh  # top left y
        y[..., 2] = x[..., 0] + dw  # bottom right x
        y[..., 3] = x[..., 1] + dh  # bottom right y
        return y
    
    def non_max_suppression(
        self,
        prediction:torch.FloatTensor,
        conf_thres:torch.FloatTensor,
        iou_thres:torch.FloatTensor,
        nc:int = 0):

        # nc = nc or (prediction.shape[1] - 4)
        bs = 1  # batch size
        nm = prediction.shape[1] - nc - 4
        mi = 4 + nc  # mask start index
        xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates
        multi_label = False
        classes = None
        agnostic = False
        max_nms=30000
        max_wh=7680
        max_det = 300


        prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
        prediction[..., :4] = self.xywh2xyxy(prediction[..., :4])  # xywh to xyxy
        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            x = x[xc[xi]]  # confidence

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Detections matrix nx6 (xyxy, conf, cls)
            box, cls, mask = x.split((4, nc, nm), 1)

            if multi_label:
                i, j = torch.where(cls > conf_thres)
                x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            else:  # best class only
                conf, j = cls.max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            i = i[:max_det]  # limit detections

            output[xi] = x[i]
        
        return output
    
    def scale_boxes(self, img1_shape:List[int], boxes:torch.Tensor, img0_shape:List[int]):
        ratio_pad = None
        padding = True
        if ratio_pad is None:
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
            pad_x = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1)
            pad_y = round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)
        else:
            gain = ratio_pad[0][0]
            pad_x = ratio_pad[1]
            pad_y = ratio_pad[2]

        # Apply padding
        if padding:
            boxes[:, [0, 2]] -= pad_x  # x padding
            boxes[:, [1, 3]] -= pad_y  # y padding

        # Normalize the boxes by the gain
        boxes[..., :4] /= gain
        self.clip_boxes(boxes, img0_shape)
        
        return boxes
    
    def clip_boxes(self,boxes:torch.Tensor, shape:List[int]):
    
        if isinstance(boxes, torch.Tensor):  # faster individually
            boxes[..., 0].clamp_(0, shape[1])  # x1
            boxes[..., 1].clamp_(0, shape[0])  # y1
            boxes[..., 2].clamp_(0, shape[1])  # x2
            boxes[..., 3].clamp_(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, shape[1])  # x1, x2
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, shape[0])  # y1, y2
    
    def process_mask(self,protos:torch.Tensor, masks_in:torch.Tensor, bboxes:torch.Tensor, shape:List[int])->torch.Tensor:

        upsample = True

        c, mh, mw = protos.shape  # CHW
        ih, iw = shape
        masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)  # CHW

        downsampled_bboxes = bboxes.clone()
        downsampled_bboxes[:, 0] *= mw / iw
        downsampled_bboxes[:, 2] *= mw / iw
        downsampled_bboxes[:, 3] *= mh / ih
        downsampled_bboxes[:, 1] *= mh / ih

        masks = self.crop_mask(masks, downsampled_bboxes)  # CHW
        if upsample:
            masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
        return masks.gt_(0.5)
    
    def crop_mask(self, masks:torch.Tensor, boxes:torch.Tensor)->torch.Tensor:

        n, h, w = masks.shape
        x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
        r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
        c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)

        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def mask_image(self, orig_img: torch.FloatTensor, boxes: List[torch.Tensor], masks: List[torch.Tensor], opacity: float = 0.5):
        class_colors = {
            0: (0,1,1),   # cyan
            1: (1,1,0),   # yellow
            2: (0,1,0),   # black
        }

        pixel_count = torch.zeros(len(self.names))

        for i in range(len(boxes)):
            masks[i] = self.clean(masks[i],boxes[i][:,-2:])
            for box, mask in zip(boxes[i], masks[i]):
                class_label = int(box[-1].item())  # Get the class label from the last element of the boxes list
                pixel_count[class_label] += torch.sum(mask)
                color = torch.tensor(class_colors.get(class_label,(0,0,0)),dtype=torch.float)  # Get the color for the class label, default to black if not found
                masked_img = orig_img.clone()
                masked_img[mask == 1] = color
                orig_img = orig_img * (1 - opacity) + masked_img * opacity
        pixel_count = [count / sum(pixel_count) for count in pixel_count]
        return orig_img, pixel_count
    
    def clean(self,masks:torch.Tensor,cls:torch.Tensor,):
        # Find the number of masks
        num_masks = masks.shape[0]
        # Iterate over each mask
        for i in range(num_masks):
            # Get the class and probability for the current mask
            current_prob,current_class = cls[i,:]
            
            # Iterate over the other masks
            for j in range(num_masks):
                if i != j:
                    # Get the class and probability for the other mask
                    other_prob,other_class = cls[j]
                    
                    # Check if the classes are different and the probabilities are lower
                    if current_prob < other_prob:
                        # Find the overlapping region between the current mask and the other mask
                        overlap = masks[i] * masks[j]
                        # Zero out the overlapping region in the current mask
                        masks[i] = masks[i] * (1 - overlap)
        
        return masks

    def pixels_count(self,masks:List[torch.Tensor], boxes: List[torch.Tensor]):

        pixel_count = torch.tensor([0,0,0],dtype=torch.float)
        
        for i in range(len(boxes)):
            for box, mask in zip(boxes[i], masks[0]):
                class_label = int(box[-1].item())  
                pixel_count[class_label] += torch.sum(mask)

        # pixel_count = [count / sum(pixel_count) for count in pixel_count]
        return pixel_count

    def save(self,path = 'yolov8.ptl'):
        self.eval()
        scripted_model = torch.jit.script(self)
        optimized_scripted_module = optimize_for_mobile(scripted_model)
        optimized_scripted_module._save_for_lite_interpreter(path)
        print(f'model saved at {path}')

    def forward(self,
                img:torch.FloatTensor,
                conf_thres: torch.FloatTensor,
                iou_thres: torch.FloatTensor,
                alpha: float = 0.6,
                ):
        
        preds = self.model(img)

        p = self.non_max_suppression(preds[0],conf_thres=conf_thres, iou_thres=iou_thres,nc = (len(self.names)))

        boxes = []
        mks = []

        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
        for i, pred in enumerate(p):
            if not len(pred):  # save empty boxes
                masks = torch.empty(0,0)
            else:
                masks = self.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:])  # HWC
                pred[:, :4] = self.scale_boxes(img.shape[2:], pred[:, :4], img.shape[2:])
            boxes.append(pred[:,:6])
            mks.append(masks)

        img,px_count = self.mask_image(img.squeeze(0).permute(1,2,0),boxes,mks)
        # px_count = self.pixels_count(mks,boxes)


        return px_count,img
    
if __name__ == '__main__':
    # model = torch.jit.load('/Users/umaidbinzubair/workspace/ai_pipline/yolov8_modified/models/yolov8m2.torchscript')
    # model = torch.jit.load('/Users/umaidbinzubair/workspace/ai_pipline/data/Training/tissue_segmentation/yolov8m/weights/best.torchscript')
    model = torch.jit.load('/Users/umaidbinzubair/workspace/ai_pipline/data/Training/tissue_noepi_segmentation/yolov8m/weights/best.torchscript')
    # model = torch.jit.load('/Users/umaidbinzubair/workspace/ai_pipline/yolov8_modified/models/tissues_yolov8m_v2.ptl')
    model = Yolov8NMS(model)
    # model.names = {0: 'Epithelium',1: 'Fibrin',2: 'Granulation',3: 'Necrosis',4: 'Epithel'} 
    model.names = {0: 'Fibrin',1: 'Granulation',2: 'Necrosis'}
    # model.save('/Users/umaidbinzubair/workspace/ai_pipline/yolov8_modified/models/tissues_yolov8m_v2_test.ptl')

    model.eval()
    # image_path = '/Users/umaidbinzubair/workspace/ai_pipline/data/11142023/images/6b4525d7-c392-47ea-b953-d8826e21.jpg'
    image_path = '/Users/umaidbinzubair/Downloads/20240322_140328.jpg'
    # image_path = '/Users/umaidbinzubair/Downloads/IMG_20240305_110452.jpg'
    # image_path = '/Users/umaidbinzubair/workspace/ai_pipline/yolov8_modified/outputs/cropped_img.jpg'
    # image_path = '/Users/umaidbinzubair/Downloads/IMG_20240305_110932.jpg'
    image = read_image(image_path)

    orig_size = image.shape[1:]
    mask_size = (640,640)
    image = T.Resize(mask_size,antialias=False)(image)
    image = image.float()
    image = image/255
    # image = image.permute(2,0,1)
    image = image.unsqueeze(0)
    conf_iou = torch.tensor(0.7,dtype = torch.float)
    conf = torch.tensor(0.2, dtype=torch.float)

    preds = model(image,conf,conf_iou)
    px_count,image = preds
    print(px_count)
    image =image.numpy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, [1200,1200])
    image = image*255
    image = image.astype(np.uint8)
    cv2.imshow('image',image)
    cv2.waitKey(0)
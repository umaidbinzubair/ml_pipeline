import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import numpy as np
from torch.utils.mobile_optimizer import optimize_for_mobile

class Yolov8NMS(nn.Module):
    def __init__(self, model:torch.jit._script.RecursiveScriptModule):
        super(Yolov8NMS, self).__init__()
        self.model = model
    
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
        iou_thres:torch.FloatTensor):

        nc = 1
        bs = 1  # batch size
        nm = 32
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
    
    def save(self,path = 'yolov8.ptl', vulkan=False, flatbuffer=False):
        self.eval()
        scripted_model = torch.jit.script(self)
        if vulkan: # vulkan backend for pytorch mobile and gpu inference
            optimized_scripted_module = optimize_for_mobile(scripted_model, backend='vulkan') # not working
        else:
            optimized_scripted_module = optimize_for_mobile(scripted_model)
        # added flatbuffer support, lowers loading time
        optimized_scripted_module._save_for_lite_interpreter(path, _use_flatbuffer=flatbuffer) 
        print(f'model saved at {path}')

    def forward(self,
                img:torch.FloatTensor,
                conf_thres: torch.FloatTensor,
                iou_thres: torch.FloatTensor,
                alpha: float = 0.6, # opacity of the mask
                margin: int = 15 # padding for crop
                ):
        
        preds = self.model(img)

        p = self.non_max_suppression(preds[0],conf_thres=conf_thres, iou_thres=iou_thres)

        results = []
        max_area = torch.tensor(0)
        max_box = torch.tensor([0, 0, 0, 0])
        mk = []
        pixels = 0
        cropped_img=img

        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
        for i, pred in enumerate(p):
            if not len(pred):  # save empty boxes
                masks = torch.empty(0,0)
            else:
                masks = self.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:])  # HWC
                pred[:, :4] = self.scale_boxes(img.shape[2:], pred[:, :4], img.shape[2:])
                mk.append(masks)

            for box in pred[:,:4]:
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area > max_area:
                    max_area = area
                    max_box = box

            results.append(pred[:,:6])
            
        if mk:

            # cropping the wound with max area
            y1 = int(max_box[1]) - margin
            y2 = int(max_box[3]) + margin
            x1 = int(max_box[0]) - margin
            x2 = int(max_box[2]) + margin
            
            # Check if the cropped region exceeds the image size
            y1 = max(0, y1)
            y2 = min(img.shape[2], y2)
            x1 = max(0, x1)
            x2 = min(img.shape[3], x2)
            
            cropped_img = cropped_img[:, :, y1:y2, x1:x2]

            stacked_mk = torch.stack(mk, dim=0)
            stacked_mk = torch.sum(stacked_mk, dim=1)
            pixels = torch.nonzero(stacked_mk).size(0)
            blue_color = torch.zeros((3, 640, 640), dtype=torch.float32)
            blue_color[2, :, :] = 1.0
            img = (1 - stacked_mk) * img + stacked_mk * (alpha * img + (1 - alpha) * blue_color)
            # img = img + (stacked_mk * blue_color)


        return results, mk, pixels, img, cropped_img
        
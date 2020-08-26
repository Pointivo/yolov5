import time
import numpy as np


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x, dtype=np.float32)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def numpy_non_max_suppression(predictions, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    if predictions.dtype == np.float:
        predictions = np.array(predictions, dtype=np.float32)

    nc = predictions[0].shape[1] - 5  # number of classes
    xc = predictions[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * predictions.shape[0]
    for xi, x in enumerate(predictions):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            # i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).t()
            i, j = np.where(x[:, 5:] > conf_thres)
            # x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            x = np.concatenate((box[i], x[i, j + 5, None], j[:, None]), 1)
        else:  # best class onl
            raise NotImplemented
            # conf, j = x[:, 5:].max(1, keepdim=True)
            # x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            # x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
            raise NotImplemented

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        # i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        i = nms(boxes=boxes, scores=scores, overlap=iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            raise NotImplemented
            # try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            #     iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            #     weights = iou * scores[None]  # box weights
            #     x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            #     if redundant:
            #         i = i[iou.sum(1) > 1]  # require redundancy
            # except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
            #     print(x, i, x.shape, i.shape)
            #     pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


# from numpy import *


# boxes is a list of size (n x 5)
# trial is a numpy array of size (n x 5)
# Author: Vicky

def nms(boxes, scores, overlap):
    if len(boxes) == 0:
        pick = []
    else:
        trial = np.zeros((len(boxes), 4), dtype=np.float64)
        trial[:] = boxes[:]
        x1 = trial[:, 0]
        y1 = trial[:, 1]
        x2 = trial[:, 2]
        y2 = trial[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        # vals = sort(scores)
        idx = np.argsort(scores)
        pick = []
        count = 1
        while idx.size != 0:
            # print "Iteration:",count
            last = idx.size
            i = idx[last - 1]
            pick.append(i)
            suppress = [last - 1]
            for pos in range(last - 1):
                j = idx[pos]
                xx1 = max(x1[i], x1[j])
                yy1 = max(y1[i], y1[j])
                xx2 = min(x2[i], x2[j])
                yy2 = min(y2[i], y2[j])
                w = xx2 - xx1 + 1
                h = yy2 - yy1 + 1
                if w > 0 and h > 0:
                    o = w * h / area[j]
                    if o > overlap:
                        suppress.append(pos)
            idx = np.delete(idx, suppress)
            count = count + 1
    return np.array(pick)

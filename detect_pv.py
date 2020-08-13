import argparse

from image_recognition.app.dvo.bbox_2d.axis_aligned_bbox_2d import AxisAlignedBbox2D
from image_recognition.app.dvo.ground_truths.object_detection import ObjectDetectionLabeledData
from models.experimental import *
from utils.datasets import *
from utils.utils import *


def detect():
    source, weights, imgsz = opt.source, opt.weights, opt.img_size

    # Initialize
    device = torch_utils.select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    dataset = LoadImages(source, img_size=imgsz)

    # Get names
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=None, agnostic=opt.agnostic_nms)
        t2 = torch_utils.time_synchronized()

        bboxes = []
        for detections in pred:
            if detections is not None and len(detections):
                detections[:, :4] = scale_coords(img.shape[2:], detections[:, :4], im0s.shape).round()
                for detected_bbox in detections:
                    detected_bbox_np = detected_bbox.cpu().numpy()
                    xmin, ymin, xmax, ymax, confidence_score, class_id = detected_bbox_np
                    class_label = names[int(class_id)]
                    bbox = AxisAlignedBbox2D(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, class_label=class_label,
                                             confidence_score=confidence_score)
                    bboxes.append(bbox)
        image_name = Path(path).name
        height, width = int(im0s.shape[0]), int(im0s.shape[1])
        od = ObjectDetectionLabeledData(image_name=image_name, bounding_boxes=bboxes, width=width, height=height)
        od.to_json_file(json_file_path=Path(path).parent / f'{Path(path).stem}.prediction.od.json')
        print(f'Saved detection file. Took {t2 - t1:0.3f} seconds.')
    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
                detect()
                strip_optimizer(opt.weights, opt.weights)
        else:
            detect()

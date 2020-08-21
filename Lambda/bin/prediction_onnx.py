from pathlib import Path

import cv2
import torch

import onnxruntime as rt
import numpy as np

from lambda_utils import numpyy_non_max_suppression

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone',
               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear',
               'hair drier', 'toothbrush']


from Lambda.bin.prediction_simple import read_image, parse_input_args, generate_visualization
from utils.utils import non_max_suppression


def get_session(model_file):
    return rt.InferenceSession(model_file)


def to_numpy(tensor):
    value = tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    return value


def main():
    opt = parse_input_args()
    image_path = Path(opt['image'])
    original_image, image_tensor = read_image(image_path=image_path, image_size=opt['img_size'])
    sess = get_session(model_file=opt['weights'])

    input_name = sess.get_inputs()[0].name
    print("input name", input_name)
    input_shape = sess.get_inputs()[0].shape
    print("input shape", input_shape)
    input_type = sess.get_inputs()[0].type
    print("input type", input_type)

    output_name = sess.get_outputs()[0].name
    print("output name", output_name)
    output_shape = sess.get_outputs()[0].shape
    print("output shape", output_shape)
    output_type = sess.get_outputs()[0].type
    print("output type", output_type)

    res = sess.run(None, {input_name: to_numpy(image_tensor)})
    batch_detections = torch.from_numpy(np.array(res[0]))
    batch_detections = non_max_suppression(batch_detections, conf_thres=0.4, iou_thres=0.5, agnostic=False)
    # pred = numpyy_non_max_suppression(prediction=batch_detections, conf_thres=0.4, iou_thres=0.5)

    image = generate_visualization(prediction=batch_detections, image_tensor=image_tensor, image=original_image,
                                   class_names=class_names)
    print(f'Result = {res}')
    if opt['view_img']:
        cv2.imshow(image_path.name, image)
        if cv2.waitKey(0) == ord('q'):  # q to quit
            raise StopIteration
    cv2.imwrite(f'/tmp/onnx_{image_path.name}', image)


if __name__ == '__main__':
    main()

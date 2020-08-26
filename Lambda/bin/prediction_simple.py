import argparse
import random
from pathlib import Path
from typing import Union, Tuple

import cv2
import numpy as np
import torch

from models.yolo import Model
from utils.datasets import letterbox
from utils.utils import non_max_suppression, scale_coords, plot_one_box
DEVICE = torch.device('cpu')


def parse_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/home/fakhir/repos/yolov5/weights/yolov5s.pt',
                        help='model.pt path(s)')
    parser.add_argument('--output', type=str, default='/tmp/output', help='output folder')  # output folder
    parser.add_argument('--image', '-i', type=str, default='/home/fakhir/repos/yolov5/inference/images/bus.jpg',
                        help='input image')  # output folder
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--view_img', action='store_true', help='display results')
    opt = parser.parse_args()
    return vars(opt)


def predict(image: np.array, model: Model):
    prediction = model(image, augment=False)
    prediction = prediction[0]
    prediction = non_max_suppression(prediction=prediction, conf_thres=0.4, iou_thres=0.5)
    return prediction


def read_image(image_path: Path, image_size: int = 640) -> Tuple[np.ndarray, torch.tensor]:
    img0 = cv2.imread(str(image_path))  # BGR
    img = letterbox(img0, new_shape=(image_size, image_size))[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(DEVICE)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img0, img


def generate_visualization(prediction, image_tensor, image, class_names):
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(class_names))]
    prediction = prediction[0]  # Only one image would be fed
    prediction[:, :4] = scale_coords(image_tensor.shape[2:], prediction[:, :4], image.shape).round()

    for *xyxy, conf, cls in prediction:
        label = '%s %.2f' % (class_names[int(cls)], conf)
        plot_one_box(xyxy, image, label=label, color=colors[int(cls)], line_thickness=3)
    return image


def main():
    opt = parse_input_args()
    image_path = Path(opt['image'])
    # original_image, image_tensor = read_image(image_path=image_path, image_size=opt['img_size'])
    original_image, image_tensor = read_image(image_path=image_path, image_size=opt['img_size'])
    model = load_model(weights_path=opt['weights'])
    prediction = predict(image=image_tensor, model=model)

    names = model.module.names if hasattr(model, 'module') else model.names

    image = generate_visualization(prediction=prediction, image_tensor=image_tensor, image=original_image,
                                   class_names=names)
    if opt['view_img']:
        cv2.imshow(image_path.name, image)
        if cv2.waitKey(0) == ord('q'):  # q to quit
            raise StopIteration
    cv2.imwrite(f'/tmp/{image_path.name}', image)


def load_model(weights_path: Union[str, Path]):
    return torch.load(weights_path, map_location=DEVICE)['model'].float().fuse().eval()


if __name__ == '__main__':
    with torch.no_grad():
        main()

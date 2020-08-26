import argparse
import ast
import base64
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from Lambda.bin.prediction_simple import read_image, generate_visualization
from Lambda.bin.prediction_onnx import class_names


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--image', default='/home/fakhir/repos/yolov5/inference/images/bus.jpg',
        help='Image on which detection is to be run')
    parser.add_argument(
        '-u', '--url', default='https://tbx8r2a16l.execute-api.us-east-1.amazonaws.com/dev/infer',
        help='Url where the lambda function is currently listening')
    parser.add_argument(
        '-s', '--save', default='/tmp/output.png',
        help='Path where the image is to be saved')
    return vars(parser.parse_args())


def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def main():
    input_args = parse_args()

    # read image
    encoded_image = get_base64_encoded_image(image_path=input_args['image'])

    # Send request
    start = time.time()
    logging.info('Sent request.')
    response = requests.post(url=input_args['url'], json=encoded_image, headers={'Content-Type': 'application/json'})
    logging.info(f'Request received in {time.time() - start} seconds')
    assert response.status_code == 200, response.content
    data_received = ast.literal_eval(response.content.decode('UTF-8'))
    batch_detections = np.asarray(json.loads(str(data_received['predictions'])))

    print(f'Result = {batch_detections}')

    # generate visualizations
    output_image, image_tensor = read_image(image_path=input_args['image'], image_size=640)

    batch_detections = torch.from_numpy(np.array(batch_detections))
    image = generate_visualization(prediction=batch_detections, image_tensor=image_tensor, image=output_image,
                                   class_names=class_names)

    image_path = Path(input_args['image'])
    cv2.imwrite(f'/tmp/onnx_lambda_{image_path.name}', image)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()

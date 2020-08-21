import base64
import io
import json
import os
import logging
from json import JSONEncoder
import time
from pathlib import Path

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.info('System modules have been loaded properly.')

import boto3
import numpy as np
import onnxruntime
from PIL import Image

# class names
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


def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def return_lambda_gateway_response(code, body):
    return {"statusCode": code, "body": json.dumps(body, cls=NumpyArrayEncoder)}


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def read_and_preprocess_image(event):
    s = int(os.environ.get('IMAGE_SIZE', 1024))
    desired_size = [s, s]
    body = event["body"]
    image_stream = io.BytesIO(base64.b64decode(body))
    # image_file = tensorflow.io.decode_image(image_stream.getvalue())
    image_file = Image.open(image_stream)
    image_file.resize(desired_size)
    logger.info(f'Image size received = {image_file.size}. Will be converted to size = {desired_size}')
    image_file = image_file.resize(desired_size)
    image_array = np.expand_dims(np.array(image_file), axis=0)
    return image_array


def init_session(weights_file_name: str):
    assert Path(weights_file_name).name == weights_file_name, \
        f'Provided weights_file_name entry is not correct. Provided = {weights_file_name}'
    s3_resource = boto3.resource('s3')
    bucket = 'yolov5-dev-serverlessdeploymentbucket-nc96dfrkc4nx'
    key = 'weights/yolov5x.onnx'
    weights_path = Path('/tmp').joinpath(weights_file_name)
    if not weights_path.exists():
        logger.info(f'Downloading weights to location = {weights_path}')
        s = time.time()
        s3_resource.Object(bucket, key).download_file(str(weights_path))
        logger.info(f'Time taken for downloading model only = {time.time() - s}')
    else:
        logger.info(f'Weights file already present. Loading from location = {weights_path}')
    return onnxruntime.InferenceSession(str(weights_path))


def get_image(event, input_size):
    # img_size_h = input_size.shape[2]
    # img_size_w = input_size.shape[3]
    img_size_h = 384
    img_size_w = 640

    # input
    body = event["body"]
    image_stream = io.BytesIO(base64.b64decode(body))
    image_src = Image.open(image_stream)
    logging.info(f'Image size received = {image_src.size}. Will be converted to size = {img_size_w},{img_size_h}')

    resized = letterbox_image(image_src, (img_size_w, img_size_h))
    img_in = np.transpose(resized, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    return img_in


try:
    session = init_session(weights_file_name='yolo5x.onnx')
except Exception as e:
    logger.exception(f'Model Loading failed = {e}')
    raise e


def detect_onnx(event, context):
    logger.info('Event Received')

    try:
        start = time.time()

        # load model
        model_load_time = time.time()
        logger.info(f'Time taken for loading models = {model_load_time - start}')

        # load image
        image = get_image(event=event, input_size=session.get_inputs()[0])
        image_load_time = time.time()
        logger.info(f'Time taken for loading image = {image_load_time - model_load_time}')

        # inference
        input_name = session.get_inputs()[0].name
        predictions = session.run(None, {input_name: image})
        predictions = predictions[0]
        logger.info(f'Time taken for prediction only = {time.time() - image_load_time}')
        return return_lambda_gateway_response(code=200,
                                              body={'predictions': predictions})
    except Exception as e:
        logger.exception(f'Error = {e}')
        error_response = {
            'error_message': "Unexpected error",
            'exception': str(e)
        }
        return return_lambda_gateway_response(503, error_response)

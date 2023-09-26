import torch
import yolov5
from yolov5.utils.loss import ComputeLoss
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
import coloredlogs
from tqdm.auto import trange

from NewPatchGen import NewPatchGen
from NewPatchGen import save_tensor_to_image

import math
import random
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import os
import sys


logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)

# category = []
# for key, value in yaml.safe_load(open('GTSRB.yaml', 'r'))['names'].items():
#     category.append(value)

category = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush']


def resolve_tuple(*args):
    return tuple(args)


OmegaConf.register_new_resolver('tuple', resolve_tuple)

# Initiate a new session


@hydra.main(config_path="configs", config_name="PatchGen")
def main(conf: DictConfig):  # conf: DictConfig
    logger.info(OmegaConf.to_yaml(conf))

    # create new folder named outputs
    import os
    isExist = os.path.exists('outputs')
    if not isExist:
        os.makedirs('outputs')

    '''
    Load Model and setup
    '''
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", autoshape=False)

    logger.info('Device: ' + str(next(model.parameters()).device))

    '''
    Attack
    '''
    learning_rate = conf.learning_rate
    batch_size = conf.batch_size
    max_iter = conf.max_iter
    patch_shape = conf.patch_shape
    patch_location = conf.patch_location
    ksize_range = conf.ksize_range
    logging_frequency = conf.logging_frequency
    rotation_range = conf.rotation_range
    tv_weight = conf.tv_weight
    skip_prob = conf.skip_prob
    scheduler_step_size = conf.scheduler_step_size
    scheduler_gamma = conf.scheduler_gamma

    if os.name == 'nt':
        # windows
        img = Image.open(
            'C:\\Users\\Devon\\Project\\NewPatchGen\\Assets\\New.png')
    elif sys.platform == 'darwin':
        # mac
        img = Image.open(
            '/Users/dayuzhang/Documents/NewPatchGen/Assets/New.png')

    img = img.rotate(180)

    import os
    isExist = os.path.exists('Patch')
    if not isExist:
        os.makedirs('Patch')

    img.save('Patch/og.jpeg')

    # add red background
    red_img = np.full((640, 640, 3), (0, 0, 255),
                      dtype=np.uint8)
    x_center = (640 - img.size[1]) // 2
    y_center = (640 - img.size[0]) // 2
    red_img[y_center:y_center + img.size[1],
            x_center:x_center + img.size[0]] = img
    red_img_reshape = red_img.transpose((2, 0, 1))
    red_image = np.stack([red_img_reshape], axis=0).astype(np.float32)

    # get contours and change all values outside of the contour to noise
    red_img_array = np.array(red_img)
    gray = cv2.cvtColor(np.array(red_img_array), cv2.COLOR_RGB2GRAY)
    edged = cv2.Canny(gray, 30, 200)
    contours, hierarchy = cv2.findContours(
        edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contours[1])

    # create mask
    mask = np.zeros_like(red_img_array)
    cv2.drawContours(mask, contours, 2, (255, 255, 255), -1)
    mask = mask[:, :, 0]

    # create 640x640 noise
    noise = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    fg = cv2.bitwise_or(red_img_array, red_img_array, mask=mask)
    mask = cv2.bitwise_not(mask)
    bk = cv2.bitwise_or(noise, noise, mask=mask)
    final = cv2.bitwise_or(fg, bk)
    cv2.imwrite('Patch/noisy.jpeg', cv2.cvtColor(final, cv2.COLOR_RGB2BGR))

    final_reshape = final.transpose((2, 0, 1))
    image = np.stack([final_reshape], axis=0).astype(np.float32)
    image = image / 255.0
    x = image.copy()

    ap = NewPatchGen(
        model=model,
        patch_shape=patch_shape,
        patch_location=patch_location,
        learning_rate=learning_rate,
        max_iter=max_iter,
        batch_size=batch_size,
        ksize_range=ksize_range,
        logging_frequency=logging_frequency,
        rotation_range=rotation_range,
        tv_weight=tv_weight,
        skip_prob=skip_prob,
        scheduler_step_size=scheduler_step_size,
        scheduler_gamma=scheduler_gamma,
    )

    patch = ap.generate(x=x)

    # add patch to the red image and then remove the red background to get back to the original image
    patch_img = Image.open('Patch/patch.png')
    x_1, y_1 = patch_location
    x_2, y_2 = x_1 + patch_shape[1], y_1 + patch_shape[2]

    red_img[x_1:x_2, y_1:y_2] = patch_img

    crop = red_img[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w]
    texture = Image.fromarray(crop.astype(np.uint8))
    texture = texture.rotate(180)
    texture.save('Patch/texture.png')

    # detect the patch
    model_pt = yolov5.load(
        'C:\\Users\\Devon\\Project\\NewPatchGen\\Assets/yolov5s.pt')
    img_path = 'Patch/final_patched.png'
    results = model_pt(img_path)

    # Render the results on the original image
    img_with_boxes = results.render()[0]
    img_with_boxes_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
    save_dir = 'Patch/final_patched_detected.png'
    cv2.imwrite(save_dir, img_with_boxes_rgb)


if __name__ == '__main__':
    main()

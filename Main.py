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

import math
import random
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING


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


@hydra.main(config_path="configs", config_name="PatchGen_copy")
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

    print(next(model.parameters()).device)

    '''
    Attack
    '''
    learning_rate = conf.learning_rate
    batch_size = conf.batch_size
    max_iter = conf.max_iter
    patch_shape = conf.patch_shape
    patch_location = conf.patch_location
    sample_size = conf.sample_size

    img = Image.open('/Users/dayuzhang/Documents/carlaAttack/Assets/New.png')
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
    x = image.copy()

    ap = NewPatchGen(
        estimator=model,
        patch_shape=patch_shape,
        patch_location=patch_location,
        learning_rate=learning_rate,
        max_iter=max_iter,
        sample_size=sample_size,
        batch_size=batch_size,
    )

    patch = ap.generate(x=x)

    # im = Image.fromarray(patch.transpose(1, 2, 0).astype(np.uint8))
    # im.save("Patch/Patch.jpeg")
    # print("Patch saved")

    # patched_image = ap.apply_patch(red_image)
    # im = Image.fromarray(patched_image[0].transpose(1, 2, 0).astype(np.uint8))
    # im.save("Patch/Patched_image.jpeg")

    # patched_image = cv2.imread("Patch/Patched_image.jpeg")
    # crop = patched_image[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w]
    # cv2.imwrite("Patch/Patched_image_crop.jpeg", crop)

    # cv2.imwrite("Patch/Patched_image_rotate.jpeg",
    #             cv2.rotate(crop, cv2.ROTATE_180))
    # print("Patched image saved")


if __name__ == '__main__':
    main()
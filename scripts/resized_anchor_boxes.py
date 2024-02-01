import os
import cv2
import multiprocessing as mp
import argparse
from itertools import repeat

def process(data):
    try:
        img_path, box, target_image_size = data
        assert os.path.exists(img_path == True), 'Image path issue.'
        x1, y1, x2, y2 = box
        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]
        target_height, target_width = target_image_size
        x1_scale = x1 / width
        y1_scale = y1 / height
        x2_scale = x2 / width
        y2_scale = y2 / height
        x1_update = x1_scale * target_width
        y1_update = y1_scale * target_height
        x2_update = x2_scale * target_width
        y2_update = y2_scale * target_height
        return [x1_update, y1_update, x2_update, y2_update]
    except:
        return f'Error: {img_path}'

def resize_anchor_boxes(img_paths, boxes, target_image_size = (256, 256), pool_size = 10):
    data = list(zip(img_paths, boxes, repeat(target_image_size)))
    with mp.Pool(pool_size) as p:
        updated_boxes = list(p.map(process, data))
    return updated_boxes
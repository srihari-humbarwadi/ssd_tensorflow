import cv2
import numpy as np

import matplotlib.pyplot as plt


def imshow(image):
    plt.figure(figsize=(8, 8))
    plt.imshow(image)


def imshow_multiple(images, titles, save_path=None):
    num_images = len(images)
    plt.subplots(1, num_images, figsize=(num_images*12, 12))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.title(titles[i])
        plt.imshow(images[i])
    if save_path is not None:
        plt.savefig(save_path)


def draw_boxes_cv2(image, boxes, categories, show_labels=True, thickness=1):
    image = np.array(image, dtype=np.uint8)
    boxes = np.array(boxes, dtype=np.int32)
    categories = np.array(categories)
    for _box, _cls in zip(boxes, categories):
        if show_labels:
            text = _cls
            char_len = len(text) * 6
            text_orig = (_box[0] + 5, _box[1] - 6)
            text_bg_xy1 = (_box[0], _box[1] - 15)
            text_bg_xy2 = (_box[0] + char_len, _box[1])
            image = cv2.rectangle(image, text_bg_xy1, text_bg_xy2, [255, 252, 150],
                                  -1)
            image = cv2.putText(image,
                                text,
                                text_orig,
                                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                .4, [0, 0, 0],
                                4,
                                lineType=cv2.LINE_AA)
            image = cv2.putText(image,
                                text,
                                text_orig,
                                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                .4, [255, 255, 255],
                                1,
                                lineType=cv2.LINE_AA)
        image = cv2.rectangle(image, (_box[0], _box[1]), (_box[2], _box[3]),
                              [30, 15, 255], thickness)
    return image

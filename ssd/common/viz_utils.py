import cv2
import numpy as np

import matplotlib.pyplot as plt


def imshow(image, figsize=(6, 6)):
    plt.figure(figsize=figsize)
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

def visualize_detections(image, boxes, classes, scores, figsize=(7, 7), linewidth=1, color=[0, 0, 1]):
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        text = '{}: {:.1f}%'.format(_cls, score * 100)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle([x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth)
        ax.add_patch(patch)
        ax.text(x1, y1, text, bbox={'facecolor':color, 'alpha':0.5})
    return ax
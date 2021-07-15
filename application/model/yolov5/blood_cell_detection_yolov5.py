import os
import sys
import time
import logging
from scipy import spatial

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.experimental import attempt_load
from utils.iou import *
from utils import image_util
from detect import predict

logger = logging.getLogger('bloodcount_logger.blood_cell_detection')

RESIZE_WIDTH = 500
RESIZE_HEIGHT = 500


def load_yolov5_model():
    model = attempt_load('./model/yolov5/weights/best_BCCM.pt')  # load .pt model
    return model


def add_borders(input_image, padding):
    cropped_image = cv2.copyMakeBorder(input_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
    return cropped_image

def resize(image_to_resize):
    dim = (RESIZE_WIDTH, RESIZE_HEIGHT)
    resized = cv2.resize(image_to_resize, dim, interpolation=cv2.INTER_AREA)
    return resized

def cut_wbc(image, coordinate, radius, zoom_out, padding):
    start_time = time.time()
    image = add_borders(image, padding)
    rect_x = (coordinate[0] + padding - (radius + zoom_out))
    rect_y = ((coordinate[1] + padding) - (radius + zoom_out))
    image = image[rect_y:(rect_y + 2 * (radius + zoom_out)), rect_x:(rect_x + 2 * (radius + zoom_out))]
    image = resize(image)
    stop_time = time.time()
    avg_time = (stop_time - start_time) * 1000
    logger.info('Cut_wbc function = ' + '{0:.5}'.format(avg_time) + 'ms')
    return image


def blood_cell_count(img_base64, model):
    logger.info('Invoked blood_cell_count')
    tic = time.time()

    rbc = 0
    wbc = 0
    platelets = 0
    counter = 0
    padding = 200

    cell = []
    cls = []
    conf = []

    record = []
    tl_ = []
    br_ = []
    iou_ = []
    iou_value = 0

    image_to_process = image_util.convert_base64_to_image(img_base64)
    predictions, names = predict(model, image_to_process)

    wbc_list = []
    for pred in predictions[0]:
        label = names[int(pred[-1])]
        confidence = pred[-2]
        tl = (pred[0], pred[1])
        br = (pred[2], pred[3])

        if label == 'RBC' and confidence < .25:
            continue
        if label == 'WBC' and confidence < .25:
            continue
        if label == 'Platelets' and confidence < .25:
            continue

        if label == 'Platelets':
            if record:
                tree = spatial.cKDTree(record)
                index = tree.query(tl)[1]
                iou_value = iou(tl + br, tl_[index] + br_[index], image_to_process)
                iou_.append(iou_value)

            if iou_value > 0.1:
                continue
            record.append(tl)
            tl_.append(tl)
            br_.append(br)

        center_x = int((tl[0] + br[0]) / 2)
        center_y = int((tl[1] + br[1]) / 2)
        center = (center_x, center_y)
        if label == 'RBC':
            color = (255, 0, 0)
            rbc = rbc + 1
        if label == 'WBC':
            color = (0, 255, 0)
            wbc = wbc + 1
        if label == 'Platelets':
            color = (0, 0, 255)
            platelets = platelets + 1
        radius = int((br[0] - tl[0]) / 2)

        cell.append([tl[0], tl[1], br[0], br[1]])
        if label == 'RBC': cls.append(0)
        if label == 'WBC': cls.append(1)
        if label == 'Platelets': cls.append(2)
        conf.append(confidence)

        if label == 'WBC':
            cut_image = cut_wbc(image_to_process, center, radius, 25, padding) #cut out wbc
            wbc_list.append(cut_image) #append to list
            counter += 1

        # augment the original image with annotations.
        font = cv2.FONT_HERSHEY_COMPLEX
        an_image = cv2.circle(image_to_process, center, radius, color, 2)
        an_image = cv2.putText(an_image, label, (center_x - 15, center_y + 5), font, .5, color, 1)

    toc = time.time()
    avg_time = (toc - tic) * 1000
    logger.info('End blood_cell_count - Time to process Annotated + cut image = ' + '{0:.5}'.format(avg_time) + 'ms')

    return an_image, wbc_list

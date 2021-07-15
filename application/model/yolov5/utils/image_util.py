import base64
import numpy as np
import cv2


def convert_image_to_base64_string(image):
    _, frame = cv2.imencode('.jpg', image)
    image_string = base64.standard_b64encode(frame).decode("utf-8")
    return image_string


def convert_base64_to_image(img_string):
    img = base64.b64decode(img_string)
    img = np.fromstring(img, dtype=np.uint8)
    return cv2.imdecode(img, 1)

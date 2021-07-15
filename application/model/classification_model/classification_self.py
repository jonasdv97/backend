import os
import time
import traceback

import numpy as np
import sys
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modelBuilder import BuildModel

logger = logging.getLogger('bloodcount_logger.classification')

class_dict = {0: ['NEUTROPHIL', 0], 1: ['EOSINOPHIL', 0], 2: ['MONOCYTE', 0], 3: ['LYMPHOCYTE', 0], 4: ['Doubtful', 0]}


def initialise_model():
    start_time = time.time()
    logger.info("Loading Classification model...")
    classification_model = BuildModel()
    classification_model.load_weights(os.path.join("model", "classification_model", "modelfinalfinal.h5"))
    stop_time = time.time()
    avg_time = (stop_time - start_time) * 1000
    logger.info('Loading Classification model complete in = ' + '{0:.5}'.format(avg_time) + 'ms')
    return classification_model


def get_classification_for_single_image(wbc_list, classification_model):
    start_time = time.time()
    logger.info('invoked get_classification_for_single_image')
    preds=[]
    for image in wbc_list:
        try:
            image_for_prediction = image.reshape((-1, 500, 500, 3))
            prediction = classification_model.predict(image_for_prediction)
            print('detected a', class_dict[np.argmax(prediction)][0].lower())
            preds.append(str(class_dict[np.argmax(prediction)][0].lower()))
        except:
            logger.error(traceback.format_exc())
            return 'Unable to classify photo'

    stop_time = time.time()
    avg_time = (stop_time - start_time) * 1000
    logger.info('get_classification_for_single_image complete in = ' + '{0:.5}'.format(avg_time) + 'ms')
    return preds

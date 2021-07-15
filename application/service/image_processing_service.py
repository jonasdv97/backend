from util.image_util import convert_image_to_base64_string
from model.yolov5.blood_cell_detection_yolov5 import load_yolov5_model, blood_cell_count
from model.classification_model.classification_self import initialise_model, get_classification_for_single_image
from model.data_transfer_model.PhotoResponseDTO import PhotoResponseDTO
import logging

classification_model = None
object_detection_model = None

logger = logging.getLogger('bloodcount_logger.image_processing_service')

def process_images_from_web_request(request_list):
    return _process_images(request_list)


def _process_images(request_list):
    global classification_model, object_detection_model
    response_list = []

    if object_detection_model is None:
        logger.debug('object_detection_model not loaded, loading now')
        object_detection_model= load_yolov5_model()
    else:
        logger.debug('tf_model was already loaded')

    if classification_model is None:
        logger.debug('bloodcount_model not loaded, loading now')
        classification_model = initialise_model()
    else:
        logger.debug('bloodcount_model was already loaded')

    for image in request_list:
        try:
            response_object = PhotoResponseDTO(image.file_name, image.original_image_base64_string, '', []) #initiate response object to return to front end
            annotated_image, wbc_list = blood_cell_count(image.original_image_base64_string, object_detection_model) #get image with annotations and a list with all extracted white blood cells
            response_object.annotated_image_base64_string = convert_image_to_base64_string(annotated_image) #convert annotated image to base64
            response_object.classification = get_classification_for_single_image(wbc_list, classification_model)
        except:
            print('ERRORRRR')
            logger.error('an error occured while processing pic: ' + image.file_name)
        else:
            logger.info('Photo successfully parsed, added to response-list')
            response_list.append(response_object.__dict__)

    logger.info('Parsing images done, returning response-list')
    return response_list

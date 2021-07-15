from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from service.image_processing_service import *
from model.data_transfer_model.PhotoRequestDTO import PhotoRequestDTO
import logging
import time

logger = logging.getLogger('bloodcount_logger')
logger.setLevel(logging.DEBUG)
log_file_handler = logging.FileHandler('logs/bloodcount.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_file_handler.setFormatter(formatter)
logger.addHandler(log_file_handler)

application = Flask(__name__)
application.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(application, resources={r"/ *": {"origins": "*"}})

@application.route('/bloodcount', methods=['POST', 'GET'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def process_images():
    global fileName, originalString, request_images_dto
    if request.method == 'POST':
        start_time = time.time()
        logger.info("POST REQUEST STARTED: ")
        if request.is_json:
            request_images_dto = []
            for obj in request.get_json():
                for key, value in obj.items():
                    if key == 'file_name':
                        fileName = value.split(".", 2)[0]
                    if key == 'original_image_base64_string':
                        originalString = "".join(value.split(",", 2)[1:])
                request_images_dto.append(PhotoRequestDTO(fileName, originalString))

        logger.info('Correctly parsed data from JSON, invoking process_images_from_web_request')
        response = process_images_from_web_request(request_images_dto)
        stop_time = time.time()
        avg_time = (stop_time - start_time) * 1000
        logger.info('POST REQUEST COMPLETED IN: ' + '{0:.5}'.format(avg_time) + 'ms')
        return jsonify(photos=response)
    elif request.method == 'GET':
        logger.info("GET DETECTED: ")
        return 'Only POST requests are allowed'
    else:
        logger.error('The request had a wrong format - returning 400')
        return "Request was not JSON", 400


@application.route('/')
def hello():
    return f'Hello from Flask!'


if __name__ == '__main__':
    application.run(debug=True, host='0.0.0.0', port=8080, use_reloader=False)

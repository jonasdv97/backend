import json
from json import JSONEncoder


class PhotoResponseDTO:
    def __init__(self, file_name='', original_image_string='', annotated_image_string='', classification=[]):
        self.file_name = file_name
        self.original_image_base64_string = original_image_string
        self.annotated_image_base64_string = annotated_image_string
        self.classification = classification

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__)

class PhotoRequestDTO:
    def __init__(self, file_name, image_string):
        self.file_name = file_name
        self.original_image_base64_string = image_string

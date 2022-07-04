from abc import ABC


class ImageSource:
    SOURCE_COUNT = 0

    def __init__(self, source_name='default_' + str(SOURCE_COUNT), source_id=SOURCE_COUNT):
        ImageSource.SOURCE_COUNT = ImageSource.SOURCE_COUNT + 1
        self.src_name = source_name
        self.id = source_id

    def set_parameter(self, param, value):
        pass

    def get_frame(self):
        pass

    def is_valid(self):
        pass



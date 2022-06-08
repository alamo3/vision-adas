from abc import ABC


class Client(ABC):

    def __init__(self, client_id):
        self.client_id = client_id

    def connect(self):
        pass

    def send_message(self, message):
        pass

    def receive_message(self, message):
        pass

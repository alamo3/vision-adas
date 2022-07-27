from mqtt.topics import *


class MessageI:

    def __init__(self, message=''):
        self.message = message

    def get_message(self):
        return self.message

    def set_message(self, message: str):
        self.message = message


class MQTTMessage(MessageI):

    def __init__(self, topic: Topic, message: str):
        super().__init__(message)
        self.topic = topic
        self.topic_str = convert_topic_string(topic)

    def get_topic(self):
        return self.topic

    def get_topic_str(self):
        return self.topic_str

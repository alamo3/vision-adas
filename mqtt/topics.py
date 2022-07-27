import enum


class Topic(enum.Enum):
    VEHICLE_GPS = 0
    LEAD_DET = 1
    TEST = 2


def convert_topic_string(topic: Topic):
    return str(topic.name)


def convert_string_topic(string: str):
    return Topic[string.upper()]

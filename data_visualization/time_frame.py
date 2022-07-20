import math


class DataPoint:

    def __init__(self, time: str, speed: float, lat: float, lon: float):
        self.time = time
        self.speed = speed
        self.lat = lat
        self.lon = lon


class ExperimentTimeFrame:

    def __init__(self, time):
        self.hour = 0
        self.minute = 0
        self.second = 0

        self.lane_change = False

        self.dp_car = []
        self.dp_bus = []

        self.hour, self.minute, self.second = self.parse_time(time)

    def parse_time(self, time_str):
        time_entries = time_str.split(':')
        hour = int(time_entries[0])
        minute = int(time_entries[1])
        second = float(time_entries[2])

        return hour, minute, second

    def is_correct_time_frame(self, time):
        time_hour, time_minute, time_second = self.parse_time(time)
        time_second_this_rounded = math.floor(self.second)
        time_second_rounded = math.floor(time_second)

        return time_hour == self.hour and time_minute == self.minute and time_second_this_rounded == time_second_rounded

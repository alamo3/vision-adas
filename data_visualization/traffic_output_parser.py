from data_visualization.time_frame import ExperimentTimeFrame
from data_visualization.time_frame import DataPoint

file_path_ego_car = '../traffic_output_ego_car.txt'
file_path_bus = '../traffic_output_bus.txt'


class TrafficOutputParser:

    def __init__(self):
        self.data = []
        self.parsed_data = []
        self.time_frames = {}
        self.parsing_ego_car = False

    def get_time_frame(self, time_str):

        time_str = self.get_key_for_time(time_str)

        if time_str in self.time_frames:
            return self.time_frames[time_str]
        else:
            time_frame = ExperimentTimeFrame(time=time_str)
            self.time_frames[time_str] = time_frame

            return time_frame

    def get_key_for_time(self, time_str):
        return time_str.split('.')[0]


    def read_file(self, file_path, ego_car):
        file = open(file_path, 'r')
        self.data = file.readlines()
        self.parsing_ego_car = ego_car

    def parse(self):
        self.parsed_data.clear()

        for entry in self.data:

            if entry.startswith('Lane'):
                self.parsed_data.append(self.add_lane_change())
            else:
                time, speed, lat, lon = self.parse_log(entry)

                time_frame = self.get_time_frame(time)

                if self.parsing_ego_car:
                    time_frame.dp_car.append(DataPoint(time, float(speed), float(lat), float(lon)))
                else:
                    time_frame.dp_bus.append(DataPoint(time, float(speed), float(lat), float(lon)))

                self.parsed_data.append(self.parse_log(entry))

        self.cleanup()

        if self.parsing_ego_car:
            self.mark_lane_changes()

    def mark_lane_changes(self):

        idx = 0
        for entry in self.parsed_data:
            if entry == 'lane_change':
                if idx == 0:
                    continue  # cannot get time data for this

                time = self.parsed_data[idx - 1][0]
                time_frame = self.get_time_frame(time)
                time_frame.lane_change = True

            idx = idx + 1


    def parse_log(self, log_line):
        entries = log_line.split(',')
        date = entries[0].split(' ')
        time = date[2]
        veh_speed = entries[5].split(' ')[3]
        lat = entries[6]
        lon = entries[7]

        lat = lat.replace(' lat:', '')
        lon = lon.replace(' lon:', '').replace('\n', '')

        return time, veh_speed, lat, lon

    def add_lane_change(self):
        return 'lane_change'

    def find_lane_change_end(self, i_start):
        i_scan = i_start + 1
        while True:
            if self.parsed_data[i_scan] == 'lane_change':
                i_scan = i_scan + 1
                continue
            else:
                return i_scan

    def cleanup(self):
        for i in range(len(self.parsed_data)):
            if self.parsed_data[i] == 'lane_change':
                i_end = self.find_lane_change_end(i)
                diff = i_end - (i + 1)

                if diff > 0:
                    del self.parsed_data[i + 1:i_end]

            if i >= len(self.parsed_data) - 1:
                break


if __name__ == "__main__":
    t_out = TrafficOutputParser()
    t_out.read_file(file_path=file_path_ego_car, ego_car=True)
    t_out.parse()
    t_out.read_file(file_path=file_path_bus, ego_car=False)
    t_out.parse()
    print('Test complete')


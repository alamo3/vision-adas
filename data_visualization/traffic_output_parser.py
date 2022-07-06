class TrafficOutputParser:

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.read_file()
        self.parsed_data = []

    def read_file(self):
        file = open(self.file_path, 'r')
        return file.readlines()

    def parse(self):
        for entry in self.data:

            if entry.startswith('Lane'):
                self.parsed_data.append(self.add_lane_change())
            else:
                self.parsed_data.append(self.parse_log(entry))

        self.cleanup()

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
    t_out = TrafficOutputParser('../traffic_output_ego_car.txt')
    t_out.parse()
    with open(r'parser_out.txt', 'w') as fp:
        fp.write("\n".join(str(item) for item in t_out.parsed_data))

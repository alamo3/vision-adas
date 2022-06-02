import threading
import copy
import atexit

import serial
from serial.tools import list_ports

GPS_VENDOR_ID = 5446
GPS_PRODUCT_ID = 423


def find_com_port_gps():
    ports = list(list_ports.comports())

    for p in ports:
        print(p)
        if p.vid == GPS_VENDOR_ID and p.pid == GPS_PRODUCT_ID:
            print('Found UBLOX GPS')
            return p

    print('Could not find UBLOX GPS!')
    return None


class GPSReceiver:

    def __init__(self):
        self.connected = False

        self.latest_df = None
        self.df_lock = threading.Lock()
        self.gps_thread = threading.Thread(target=self.gps_serial_parser, daemon=True)
        self.exit_thread = False

        self.connected_port = None
        self.gps_com = None

        atexit.register(self.terminate_gps)

    def terminate_gps(self):
        self.exit_thread = True

    def connect(self):
        port = find_com_port_gps()

        assert port is not None
        self.gps_com = serial.Serial(port=port.name, baudrate=9600)
        self.connected_port = port.name
        self.connected = True

        self.gps_thread.start()

    def get_data_frame(self):
        self.df_lock.acquire()
        df_return = copy.copy(self.latest_df)
        self.df_lock.release()
        return df_return

    def get_valid_serial_data(self):
        gps_data = self.gps_com.readline().decode('utf-8').split(",")
        while gps_data[0] != '$GPRMC':
            gps_data = self.gps_com.readline().decode('utf-8').split(",")

        return gps_data

    def parse_lat(self, lat_ddmm):
        deg = float(lat_ddmm[0:2])
        minutes = float(lat_ddmm[2:])
        seconds = minutes / 60
        lat = deg + seconds
        return lat

    def parse_lon(self, lon_dddmm):
        deg = float(lon_dddmm[0:3])
        minutes = float(lon_dddmm[3:])
        seconds = minutes / 60
        lon = deg + seconds
        return lon

    def gps_serial_parser(self):

        while not self.exit_thread:
            try:
                print('running')
                gps_data = self.get_valid_serial_data()

                lat_ddmm = gps_data[3]
                lat = self.parse_lat(lat_ddmm)

                if gps_data[4] == 'S':
                    lat = -lat  # southern latitudes are negative

                lon_dddmm = gps_data[5]
                lon = self.parse_lon(lon_dddmm)

                if gps_data[6] == 'W':
                    lon = -lon  # western longitudes are negative

                spd_knots = float(gps_data[7])
                spd_ms = spd_knots / 1.944

                self.df_lock.acquire()
                gps_df = {'lat': lat, 'lon': lon, 'speed': spd_ms, 'speed_knots': spd_knots}
                self.latest_df = gps_df
                self.df_lock.release()

            except Exception as e:
                print('GPS error!')
                print(e)

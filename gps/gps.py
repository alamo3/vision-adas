import threading
import copy
import atexit

import serial
from serial.tools import list_ports

GPS_VENDOR_ID = 5446
GPS_PRODUCT_ID = 423


def find_com_port_gps():
    """
    Finds GPS com USB port based on usb product and vendor ID defined above.
    :return: Port
    """
    ports = list(list_ports.comports())

    for p in ports:
        print(p)
        if p.vid == GPS_VENDOR_ID and p.pid == GPS_PRODUCT_ID:
            print('Found UBLOX GPS')
            return p

    print('Could not find UBLOX GPS!')
    return None


class GPSReceiver:
    """
    GPSReceiver continuously receives data from gps receiver in the background and parses
    it into usable values in the client program.
    Currently parses: latitude, longitude (decimal degrees), speed (m/s), speed (knots).
    """
    def __init__(self):
        self.connected = False

        # setup data variables
        self.latest_df = None

        # we will be using threads to run information parser in the background
        self.df_lock = threading.Lock()  # data lock to prevent undefined behaviour
        self.gps_thread = threading.Thread(target=self.gps_serial_parser, daemon=True)
        self.exit_thread = False

        self.connected_port = None
        self.gps_com = None

        atexit.register(self.terminate_gps)

    def terminate_gps(self):
        self.exit_thread = True

    def connect(self):
        """
        Attempts to connect to GPS receiver. Will fail if no GPS receiver is connected.
        Once connected, automatically starts data parsing thread.
        :return: None
        """
        port = find_com_port_gps()

        assert port is not None
        self.gps_com = serial.Serial(port=port.device, baudrate=9600)
        self.connected_port = port.name
        self.connected = True

        self.gps_thread.start()

    def get_data_frame(self):
        self.df_lock.acquire()
        df_return = copy.copy(self.latest_df)
        self.df_lock.release()
        return df_return

    def get_valid_serial_data(self):
        """
        Parses serial data based on the NMEA sentences format.
        See: https://www.rfwireless-world.com/Terminology/GPS-sentences-or-NMEA-sentences.html
        We only care about the $GPRMC sentence, so we keep reading till that sentence is
        sent by GPS Receiver. Roughly sent once a second by GPS receiver
        :return: GPRMC sentence when available
        """
        gps_data = self.gps_com.readline().decode('utf-8').split(",")
        while gps_data[0] != '$GPRMC':
            gps_data = self.gps_com.readline().decode('utf-8').split(",")

        return gps_data

    # Convert arc minute do decimal degree coordinates
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
                gps_data = self.get_valid_serial_data()  # get GPRMC sentence

                # parse data into decimal degree coordinates
                lat_ddmm = gps_data[3]
                lat = self.parse_lat(lat_ddmm)

                if gps_data[4] == 'S':
                    lat = -lat  # southern latitudes are negative

                lon_dddmm = gps_data[5]
                lon = self.parse_lon(lon_dddmm)

                if gps_data[6] == 'W':
                    lon = -lon  # western longitudes are negative

                # speed received is in knots. convert to m/s
                spd_knots = float(gps_data[7])
                spd_ms = spd_knots / 1.944

                # update latest data frame
                self.df_lock.acquire()
                gps_df = {'lat': lat, 'lon': lon, 'speed': spd_ms, 'speed_knots': spd_knots}
                self.latest_df = gps_df
                self.df_lock.release()

            except Exception as e:
                print('GPS error!')
                print(e)

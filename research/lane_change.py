from datetime import datetime
from playsound import playsound
import geopy.distance as gd
from gps import *
from main import out_traffic
from main import field_experiment
gtim = 96  # green time (with 120 sec cycle)
tcyc = 120  # cycle length is 120 sec
tdwl = 30  # bus dwelling time
dbus = 20  # distance between bus station and the signal

gpsd = None

if field_experiment:
    gpsd = gps(mode=WATCH_ENABLE|WATCH_NEWSTYLE)


def lane_change_algo(b_dist, speed):
    nx = gpsd.next()
    if nx['class'] == 'TPV':
        lat = getattr(nx, 'lat', "Unknown")
        lon = getattr(nx, 'lon', "Unknown")
        lane_change_algo_lat_lon(b_dist, speed, lat, lon)


def lane_change_algo_lat_lon(b_dist, speed, lat, lon):
    coor1 = (lat, lon)
    coor0 = (-79.930346333, 43.261875833)  # ---------------signal coordination, to be modified ---------------#
    s_dist = gd.distance(coor0, coor1).km * 1000
    s_vel = speed

    # sveh=15/3.6 # speed of individual vehicles
    # s_dist=s_dist+dbus

    if s_dist - b_dist > dbus:  # the bus is not stopping yet
        dateTimeObj = datetime.now()
        tcur = int((dateTimeObj.minute % 2) * 60 + dateTimeObj.second)

        # estimate the expected time when the car pass the intersection without bus stop
        tim0 = s_dist / s_vel + tcur
        tim1 = tim0 % tcyc
        if tim1 < gtim:
            timc0 = tim0
        else:
            timc0 = tim0 - tim1 + tcyc

        # estimate the expected time when the car pass the intersection with bus stop
        tmp0 = s_dist / s_vel + tdwl + tcur
        tmp1 = tmp0 % tcyc
        if tmp1 < gtim:
            timc1 = tmp0
        else:
            timc1 = tmp0 - tmp1 + tcyc

        if timc0 + 4.0 < timc1 and tcur % 5 == 0:
            notify_driver()  # play lane changing instruction

            info = "Lane Changing Instruction is Generated" + "\n"
            out_traffic.write(info)



def notify_driver():
    playsound('LCAudio.mp3')

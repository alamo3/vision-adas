import time

from rpi_client import RPIClient
#from gps.gps import GPSReceiver

# Unit test for MQTT Communication test
class Connection():

    def __init__(self, message):
        self.message = message

    def publish(self):

        rpi_client = RPIClient()
        rpi_client.connect()
        time.sleep(5)
        rpi_client.send_message('car/vehspeed;'+self.message)


    def subscriber(self):
        rpi_client = RPIClient()

        rpi_client.connect()
        time.sleep(5)
        rpi_client.subscribe(topic='car/vehspeed')
        time.sleep(5)



#if __name__ == '__main__':
    #unittest.main()
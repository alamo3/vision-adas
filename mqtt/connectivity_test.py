import time
import unittest

from mqtt.topics import Topic
from mqtt.message import MQTTMessage
from mqtt_client import MQTTClient
from gps.gps import GPSReceiver


# Unit test for MQTT Communication test
class MyTestCase(unittest.TestCase):
    def test_connection(self):
        client_1 = MQTTClient(client_id='VEHICLE-1')

        client_2 = MQTTClient(client_id='VEHICLE-2')

        client_1.connect()

        client_2.connect()

        client_1.subscribe(topic=Topic.LEAD_DET)

        client_2.send_message(MQTTMessage(topic=Topic.LEAD_DET, message='Lead,2.3'))

        time.sleep(3)

        self.assertEqual('Lead,2.3', client_1.last_message)

    def test_connection_publish_1(self):

        publish_vehicle = MQTTClient(client_id='VEHICLE-1')

        publish_vehicle.connect()

        publish_vehicle.send_message(MQTTMessage(topic=Topic.TEST, message='abc123'))

        time.sleep(5)

    def test_connection_subscriber_1(self):

        NUM_SUBSCRIBERS = 5

        subscribers = []

        for i in range(NUM_SUBSCRIBERS):
            sub_vehicle = MQTTClient(client_id='VEHICLE-' + str(i + 2))
            sub_vehicle.connect()
            sub_vehicle.subscribe(topic=Topic.TEST)

            subscribers.append(sub_vehicle)

        print('Ready to receive')

        time.sleep(30)

        for sub in subscribers:
            self.assertEqual('abc123', sub.last_message)
            print(sub.client_id, ' OK')

        print('All OK')


if __name__ == '__main__':
    unittest.main()

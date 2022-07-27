import time
import unittest

from mqtt.topics import Topic
from mqtt.message import MQTTMessage
from mqtt_client import MQTTClient


# Unit test for MQTT Communication test
class MyTestCase(unittest.TestCase):

    """
    This test case tests a simple connection between one client and another. It simulates one vehicle publishing
    its own lead detection data to the MQTT broker/server and another vehicle receiving it. We confirm proper
    receiving at the end
    """
    def test_connection(self):

        # create 2 vehicles
        client_1 = MQTTClient(client_id='VEHICLE-1')

        client_2 = MQTTClient(client_id='VEHICLE-2')

        # connect both of them
        client_1.connect()

        client_2.connect()

        # subscribe the first vehicle to lead detection topic

        client_1.subscribe(topic=Topic.LEAD_DET)

        # second vehicle publishes message to lead detection topic
        # (Note: Do not need to sub to publish message on topic)

        client_2.send_message(MQTTMessage(topic=Topic.LEAD_DET, message='Lead,2.3'))

        time.sleep(3)  # give time for message to arrive (this is too high, want to test with lower time).

        self.assertEqual('Lead,2.3', client_1.last_message)

    """
    Test case is intended to be used in conjunction with test_connection_subscriber_1. The two test cases in combination 
    try to simulate a one to many communication scenario instead of simple 1-1. Many subscribing vehicles are
    simulated in test_connection_subscriber_1. We run that test first and wait till all vehicles successfully connect
    to the server. Then we run test_connection_publish_1 to send a message to the TEST topic. If successful all vehicles
    in test_connection_subscriber_1 should receive the message published from this unit test. Additionally, we also
    confirm to see if the test successfully published the message at the end. This helps with figuring out where
    any issue is happening.
    """
    def test_connection_publish_1(self):

        # create publishing vehicle and connect it
        publish_vehicle = MQTTClient(client_id='VEHICLE-1')

        publish_vehicle.connect()

        # send message and confirm if it was sent
        publish_vehicle.send_message(MQTTMessage(topic=Topic.TEST, message='abc123'))

        time.sleep(5)
        self.assertEqual(True, publish_vehicle.last_publish_successful)

    def test_connection_subscriber_1(self):

        NUM_SUBSCRIBERS = 5  # Control the number of receiving vehicles simulated

        subscribers = []

        # Create, connect and subscribe all receiving vehicles
        for i in range(NUM_SUBSCRIBERS):
            sub_vehicle = MQTTClient(client_id='VEHICLE-' + str(i + 2))
            sub_vehicle.connect()
            sub_vehicle.subscribe(topic=Topic.TEST)

            subscribers.append(sub_vehicle)

        print('Ready to receive')

        time.sleep(30) # Wait for message to arrive (This time is too high, need to find lower bound)

        # Confirm message is received by all subscribers (Run test_connection_publish_1)
        for sub in subscribers:
            self.assertEqual('abc123', sub.last_message)
            print(sub.client_id, ' OK')

        print('All OK')


if __name__ == '__main__':
    unittest.main()

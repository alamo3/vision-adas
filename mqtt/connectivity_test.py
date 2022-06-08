import time
import unittest

from rpi_client import RPIClient

# Unit test for MQTT Communication test
class MyTestCase(unittest.TestCase):
    def test_connection(self):
        rpi_client = RPIClient()

        rpi_client2 = RPIClient()

        self.assertEqual(rpi_client.client_id, 'RPI-0')
        self.assertEqual(rpi_client2.client_id, 'RPI-1')

        rpi_client.connect()

        rpi_client2.connect()

        rpi_client.subscribe(topic='car/vehspeed')

        rpi_client2.send_message('car/vehspeed;20')

        time.sleep(3)

        self.assertEqual(str(rpi_client.last_message), 'b\'20\'')


if __name__ == '__main__':
    unittest.main()

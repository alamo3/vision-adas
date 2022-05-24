from mqtt_client import MQTTClient


class RPIClient(MQTTClient):

    num_clients = 0

    def __init__(self):
        MQTTClient.__init__(self, client_id='RPI-'+str(RPIClient.num_clients))

        RPIClient.num_clients = RPIClient.num_clients + 1

        self.last_message = ''

    def send_message(self, message, qos=1):
        super().send_message(message, qos)

    def receive_message(self, message):
        msg = message.split(";")
        self.last_message = msg[1]




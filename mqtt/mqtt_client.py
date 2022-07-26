from base_client import Client
from mqtt.topics import Topic

import paho.mqtt.client as paho
from paho import mqtt

import logging
import os

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

# Details for MQTT HiveMQ server
USER_NAME = 'mcmaster'
PASSWORD = 'McMaster123'
SERVER_URL = 'ae7660133e3d4822897f1256213846b0.s1.eu.hivemq.cloud'
SERVER_PORT = 8883

WEBSOCKET_PORT = 8884


class MQTTClient(Client):
    """
    Provides base implementation for an MQTT Client. Extends Client class
    We can connect to an MQTT server, subscribe, send and publish messages for topics
    using this class.
    """

    def __init__(self, client_id):
        Client.__init__(self, client_id)

        self.logger = logging.getLogger(__name__)

        self.client = paho.Client(client_id=client_id, userdata=None, protocol=paho.MQTTv5)
        self.last_message = ''

    def on_connect(self, client, userdata, flags, rc, properties=None):
        print("CONNACK received with code %s." % rc)

    # with this callback you can see if your publish was successful
    def on_publish(self, client, userdata, mid, properties=None):
        print("mid: " + str(mid))

    # print which topic was subscribed to
    def on_subscribe(self, client, userdata, mid, granted_qos, properties=None):
        print("Subscribed: " + str(mid) + " " + str(granted_qos))

    # print message, useful for checking if it was successful
    def on_message(self, client, userdata, msg):
        message_dict = {'topic' : str(msg.topic), 'message' : str(msg.payload)}
        self.receive_message(message_dict)
        print(msg.topic + " " + str(msg.qos) + " " + str(msg.payload))

    def connect(self):

        # we are using SSL encryption to connect
        self.client.on_connect = self.on_connect
        self.client.tls_set(tls_version=mqtt.client.ssl.PROTOCOL_TLS)

        self.client.username_pw_set(username=USER_NAME, password=PASSWORD)

        # connect and start non-blocking loop. See Paho MQTT docs for details
        self.client.connect(SERVER_URL, SERVER_PORT)
        self.client.loop_start()

        # set callback functions
        self.client.on_subscribe = self.on_subscribe
        self.client.on_message = self.on_message
        self.client.on_publish = self.on_publish

    def subscribe(self, topic, qos=1):
        self.client.subscribe(topic, qos=qos)

    def send_message(self, message, qos=1):
        self.client.publish(message['topic'], message['message'], qos=qos)

    def receive_message(self, message):
        super().receive_message(message)
        self.last_message = message['message']







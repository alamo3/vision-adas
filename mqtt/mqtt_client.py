from base_client import Client
from mqtt.topics import *
from mqtt.message import MQTTMessage

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

DEBUG = True


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
        if DEBUG:
            print(self.client_id, "CONNACK received with code %s." % rc)

    # with this callback you can see if your publish was successful
    def on_publish(self, client, userdata, mid, properties=None):
        if DEBUG:
            print(self.client_id, 'Successfully published message')

    # print which topic was subscribed to
    def on_subscribe(self, client, userdata, mid, granted_qos, properties=None):
        if DEBUG:
            print(self.client_id, "Subscribed: " + str(mid) + " " + str(granted_qos))

    # print message, useful for checking if it was successful
    def on_message(self, client, userdata, msg):
        message = MQTTMessage(topic=convert_string_topic(str(msg.topic)), message=str(msg.payload)[2:-1])
        self.receive_message(message)

        if DEBUG:
            print(self.client_id, msg.topic + " " + str(msg.payload))

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

    def subscribe(self, topic: Topic, qos=1):
        self.client.subscribe(convert_topic_string(topic), qos=qos)

    def send_message(self, message: MQTTMessage, qos=1):
        self.client.publish(message.get_topic_str(), message.get_message(), qos=qos)

    def receive_message(self, message: MQTTMessage):
        self.last_message = message.get_message()

"""
MQTT broker connectivity test (local Mosquitto by default).

Run:  python -m src.test_mqtt_connection

Publishes a test message and exits. Does not use the camera.
Note: faceLockServo.py uses broker 157.173.101.159 — change host below
      to match src/config.py if testing the production broker.
"""

import time

import paho.mqtt.client as mqtt


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("SUCCESS: Connected to Mosquitto broker!")
        print("   Broker: localhost:1883")
    else:
        print(f"Failed to connect. Return code: {rc}")


def on_disconnect(client, userdata, rc):
    print("Disconnected from broker")


# MQTT client setup
client = mqtt.Client()
client.on_connect = on_connect
client.on_disconnect = on_disconnect

print("Attempting to connect to Mosquitto...")
try:
    client.connect("localhost", 1883, 60)
    client.loop_start()

    time.sleep(2)

    # Test publish to a sample topic
    result = client.publish(
        "TeAmOnE/facelocking/servo_ctrl_x9z",
        "Connection test successful",
    )
    if result.rc == mqtt.MQTT_ERR_SUCCESS:
        print("SUCCESS: Published test message")
    else:
        print("Failed to publish message")

    time.sleep(1)

finally:
    client.loop_stop()
    client.disconnect()

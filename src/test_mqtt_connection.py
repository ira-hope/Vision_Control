import paho.mqtt.client as mqtt
import time

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ SUCCESS: Connected to Mosquitto broker!")
        print(f"   Broker: localhost:1883")
    else:
        print(f"❌ Failed to connect. Return code: {rc}")

def on_disconnect(client, userdata, rc):
    print("Disconnected from broker")

# Create client
client = mqtt.Client()
client.on_connect = on_connect
client.on_disconnect = on_disconnect

# Try to connect
print("Attempting to connect to Mosquitto...")
try:
    client.connect("localhost", 1883, 60)
    client.loop_start()
    
    # Keep connection alive for a few seconds
    time.sleep(2)
    
    # Test publish
    result = client.publish("TeAmOnE/facelocking/servo_ctrl_x9z", "Connection test successful")
    if result.rc == mqtt.MQTT_ERR_SUCCESS:
        print("✅ SUCCESS: Published test message")
    else:
        print("❌ Failed to publish message")
        
    time.sleep(1)
    
finally:
    client.loop_stop()
    client.disconnect()
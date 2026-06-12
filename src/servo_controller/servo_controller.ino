/*
 * ESP8266 MQTT Servo Controller for Face Locking project
 *
 * Subscribes to MQTT topic (see topic_servo_angle below) and moves a servo
 * on pin D1 to the received angle (0-180 degrees).
 *
 * Search mode: if no MQTT message for SEARCH_TIMEOUT ms, servo sweeps back
 * and forth automatically until tracking resumes.
 *
 * Flash with Arduino IDE + ESP8266 board support.
 * Libraries: ESP8266WiFi, PubSubClient, Servo
 *
 * Must match MQTT_TOPIC_SERVO_ANGLE in src/config.py (faceLockServo.py).
 */

#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <Servo.h>

// --- WiFi credentials (change for your network) ---
const char* ssid = "Main Hall";
const char* password = "Meeting@2024";

// --- MQTT broker (must match src/config.py MQTT_BROKER / MQTT_PORT) ---
const char* mqtt_server = "157.173.101.159";
const int mqtt_port = 1883;
const char* topic_servo_angle = "TeAmSiX/facelocking/servo_ctrl_x9z";

const int SERVO_PIN = D1;

WiFiClient espClient;
PubSubClient client(espClient);
Servo myServo;
int currentAngle = 90;   // actual position written to servo each loop
int targetAngle = 90;    // desired position from MQTT or search mode

// Search mode: sweep when Python stops sending angles
unsigned long lastMessageTime = 0;
const unsigned long SEARCH_TIMEOUT = 1500;   // ms without message before search
int searchDirection = 1;
const int SEARCH_STEP = 2;

void setup() {
  Serial.begin(9600);
  Serial.println("\n===== Servo Controller =====");

  myServo.attach(SERVO_PIN);
  myServo.write(currentAngle);
  delay(1000);

  connectToWiFi();
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);

  Serial.println("Ready to receive angles...");
}

void connectToWiFi() {
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi connected!");
}

// Called when an MQTT message arrives on a subscribed topic
void callback(char* topic, byte* payload, unsigned int length) {
  char message[length + 1];
  memcpy(message, payload, length);
  message[length] = '\0';

  if (String(topic) == topic_servo_angle) {
    int newAngle = atoi(message);

    if (newAngle >= 0 && newAngle <= 180) {
      targetAngle = newAngle;
      lastMessageTime = millis();   // reset search-mode timer
      Serial.print("Target angle: ");
      Serial.println(targetAngle);
    }
  }
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("Connecting to MQTT...");
    if (client.connect("servo_controller")) {
      Serial.println("connected!");
      client.subscribe(topic_servo_angle);
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" retrying in 5 sec...");
      delay(5000);
    }
  }
}

void loop() {
  if (!client.connected()) reconnect();
  client.loop();

  unsigned long now = millis();

  // No recent MQTT -> sweep servo to search for the face
  if (now - lastMessageTime > SEARCH_TIMEOUT) {
    targetAngle += SEARCH_STEP * searchDirection;

    if (targetAngle >= 180 || targetAngle <= 0) {
      searchDirection *= -1;
    }

    targetAngle = constrain(targetAngle, 0, 180);
  }

  // Move currentAngle toward targetAngle by 1 degree per tick
  if (currentAngle < targetAngle) currentAngle++;
  else if (currentAngle > targetAngle) currentAngle--;

  myServo.write(currentAngle);

  delay(15);
}

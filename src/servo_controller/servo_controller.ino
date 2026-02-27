#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <Servo.h>

const char* ssid = "Main Hall";
const char* password = "Meeting@2024";

const char* mqtt_server = "157.173.101.159";  
const int mqtt_port = 1883;
const char* topic_servo_angle = "TeAmSiX/facelocking/servo_ctrl_x9z";

const int SERVO_PIN = D1;

WiFiClient espClient;
PubSubClient client(espClient);
Servo myServo;
int currentAngle = 90;
int targetAngle = 90;

// 🔎 Search Mode Settings
unsigned long lastMessageTime = 0;
const unsigned long SEARCH_TIMEOUT = 1500;   // ms without message before search starts
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

void callback(char* topic, byte* payload, unsigned int length) {
  char message[length + 1];
  memcpy(message, payload, length);
  message[length] = '\0';

  if (String(topic) == topic_servo_angle) {
    int newAngle = atoi(message);

    if (newAngle >= 0 && newAngle <= 180) {
      targetAngle = newAngle;
      lastMessageTime = millis();   // 🔥 reset timeout
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

  // 🔎 If no MQTT message for some time → SEARCH MODE
  if (now - lastMessageTime > SEARCH_TIMEOUT) {
    targetAngle += SEARCH_STEP * searchDirection;

    if (targetAngle >= 180 || targetAngle <= 0) {
      searchDirection *= -1;  // reverse direction
    }

    targetAngle = constrain(targetAngle, 0, 180);
  }

  // Smooth movement
  if (currentAngle < targetAngle) currentAngle++;
  else if (currentAngle > targetAngle) currentAngle--;

  myServo.write(currentAngle);

  delay(15);
}
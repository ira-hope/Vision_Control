/*
 * ESP8266 MQTT Servo Controller for Face Locking (BENAX assessment)
 *
 * Subscribes to MQTT topic (see topic_motor_cmd below) and drives a servo on
 * pin D5 (GPIO14). Accepts assessment motor commands:
 *   MOVED_LEFT, MOVED_RIGHT, CENTERED, STOPPED, OUT_OF_FRAME, SCAN
 *
 * SCAN / OUT_OF_FRAME  -> sweep 0-180 degrees back and forth
 * MOVED_LEFT           -> pan left  (+PAN_STEP degrees)
 * MOVED_RIGHT          -> pan right (-PAN_STEP degrees)
 * CENTERED / STOPPED   -> hold current angle
 *
 * Legacy: numeric strings "0"-"180" set an absolute angle.
 *
 * Flash with Arduino IDE + ESP8266 board support.
 * Libraries: ESP8266WiFi, PubSubClient, Servo
 *
 * Must match MQTT_TOPIC_MOTOR_CMD in src/config.py.
 */

#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <Servo.h>

// --- WiFi credentials (change for your network) ---
const char* ssid = "Chrono";
const char* password = "chrono@2026";

// --- MQTT broker (must match src/config.py) ---
const char* mqtt_server = "157.173.101.159";
const int mqtt_port = 1883;
const char* topic_motor_cmd = "TeAmSiX/facelocking/servo_ctrl_x9z";

const int SERVO_PIN = D5;  // GPIO14 on NodeMCU / Wemos D1 mini

WiFiClient espClient;
PubSubClient client(espClient);
Servo myServo;

int currentAngle = 90;
int targetAngle = 90;
bool scanMode = true;
int searchDirection = 1;

unsigned long lastMessageTime = 0;
unsigned long lastTrackTime = 0;
const unsigned long SEARCH_TIMEOUT = 1500;  // ms without MQTT -> auto SCAN
const int PAN_STEP = 2;       // degrees per MOVED_LEFT / MOVED_RIGHT
const int SEARCH_STEP = 2;    // degrees per loop tick while scanning
const int TRACK_STEP = 1;     // degrees per tracking tick toward MQTT target
const int TRACK_INTERVAL_MS = 30;  // slower ramp = smoother follow
const int LOOP_DELAY_MS = 12;

bool isNumericAngle(const char* msg, int* outAngle) {
  if (msg == nullptr || msg[0] == '\0') {
    return false;
  }
  for (const char* p = msg; *p != '\0'; p++) {
    if (!isDigit(*p)) {
      return false;
    }
  }
  int angle = atoi(msg);
  if (angle < 0 || angle > 180) {
    return false;
  }
  *outAngle = angle;
  return true;
}

void writeServo(int angle) {
  currentAngle = constrain(angle, 0, 180);
  myServo.write(currentAngle);
}

void setTargetAngle(int angle) {
  targetAngle = constrain(angle, 0, 180);
}

void rampTowardTarget() {
  unsigned long now = millis();
  if (now - lastTrackTime < (unsigned long)TRACK_INTERVAL_MS) {
    return;
  }
  lastTrackTime = now;

  if (currentAngle < targetAngle) {
    currentAngle += TRACK_STEP;
  } else if (currentAngle > targetAngle) {
    currentAngle -= TRACK_STEP;
  } else {
    return;
  }

  currentAngle = constrain(currentAngle, 0, 180);
  myServo.write(currentAngle);
}

void enterScanMode(const char* reason) {
  scanMode = true;
  Serial.print("SCAN mode (");
  Serial.print(reason);
  Serial.print(") angle=");
  Serial.println(currentAngle);
}

void applyMotorCommand(const char* msg) {
  lastMessageTime = millis();

  int numericAngle = 0;
  if (isNumericAngle(msg, &numericAngle)) {
    bool exitingScan = scanMode;
    scanMode = false;
    setTargetAngle(numericAngle);
    if (exitingScan) {
      writeServo(numericAngle);
      Serial.print("Snap after SCAN: ");
    } else {
      Serial.print("Target angle: ");
    }
    Serial.println(targetAngle);
    return;
  }

  if (strcmp(msg, "MOVED_LEFT") == 0) {
    scanMode = false;
    setTargetAngle(currentAngle + PAN_STEP);
    Serial.print("MOVED_LEFT target: ");
    Serial.println(targetAngle);
    return;
  }

  if (strcmp(msg, "MOVED_RIGHT") == 0) {
    scanMode = false;
    setTargetAngle(currentAngle - PAN_STEP);
    Serial.print("MOVED_RIGHT target: ");
    Serial.println(targetAngle);
    return;
  }

  if (strcmp(msg, "CENTERED") == 0 || strcmp(msg, "STOPPED") == 0) {
    scanMode = false;
    Serial.print("HOLD at ");
    Serial.println(currentAngle);
    return;
  }

  if (strcmp(msg, "SCAN") == 0 || strcmp(msg, "OUT_OF_FRAME") == 0) {
    enterScanMode(msg);
    return;
  }

  Serial.print("Unknown command: ");
  Serial.println(msg);
}

void callback(char* topic, byte* payload, unsigned int length) {
  char message[length + 1];
  memcpy(message, payload, length);
  message[length] = '\0';

  if (String(topic) == topic_motor_cmd) {
    applyMotorCommand(message);
  }
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

void reconnect() {
  while (!client.connected()) {
    Serial.print("Connecting to MQTT...");
    if (client.connect("servo_controller")) {
      Serial.println("connected!");
      client.subscribe(topic_motor_cmd);
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" retrying in 5 sec...");
      delay(5000);
    }
  }
}

void runScanSweep() {
  currentAngle += SEARCH_STEP * searchDirection;

  if (currentAngle >= 180) {
    currentAngle = 180;
    searchDirection = -1;
  } else if (currentAngle <= 0) {
    currentAngle = 0;
    searchDirection = 1;
  }

  writeServo(currentAngle);
}

void setup() {
  Serial.begin(9600);
  Serial.println("\n===== Servo Controller (motor commands) =====");

  myServo.attach(SERVO_PIN);
  writeServo(currentAngle);
  delay(500);

  connectToWiFi();
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);

  lastMessageTime = millis();
  enterScanMode("startup");
  Serial.println("Ready for MOVED_LEFT / MOVED_RIGHT / SCAN ...");
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  unsigned long now = millis();

  if (now - lastMessageTime > SEARCH_TIMEOUT) {
    if (!scanMode) {
      enterScanMode("mqtt timeout");
    }
  }

  if (scanMode) {
    runScanSweep();
    targetAngle = currentAngle;
  } else {
    rampTowardTarget();
  }

  delay(LOOP_DELAY_MS);
}

// // calib_arduino.ino
// #include <Arduino.h>

// // ---------- Motor pins ----------
// #define STEP_WHEEL 2
// #define DIR_WHEEL 3
// #define STEP_BASE 8
// #define DIR_BASE 5
// #define ENABLE_PIN 6

// // ---------- Encoder pins (set according to wiring) ----------
// #define ENC_BASE_A 18   // NOTE: If using pin 2 for encoder A, avoid conflict with step pins - change pins as per wiring
// #define ENC_BASE_B 19
// #define ENC_WHEEL_A 20
// #define ENC_WHEEL_B 21

// // For reliability, choose interrupt-capable pins for A channels (2,3 on many Arduinos).
// // You may need to move step pins if conflicts arise. Adjust wiring as needed.

// // ---------- Encoder counts to degrees ----------
// #define COUNTS_PER_REV_BASE 800  // change to your encoder CPR
// #define GEAR_RATIO_BASE 1.0      // set if gearbox
// #define COUNTS_PER_DEGREE_BASE ((COUNTS_PER_REV_BASE * GEAR_RATIO_BASE) / 360.0)

// #define COUNTS_PER_REV_WHEEL 800
// #define GEAR_RATIO_WHEEL 1.0
// #define COUNTS_PER_DEGREE_WHEEL ((COUNTS_PER_REV_WHEEL * GEAR_RATIO_WHEEL) / 360.0)

// // ---------- Motion params ----------
// #define STEPS_PER_DEGREE_BASE 132
// #define STEPS_PER_DEGREE_WHEEL 132
// #define MAX_BASE_DEG 90
// #define MIN_BASE_DEG -90
// #define MAX_WHEEL_DEG 45
// #define MIN_WHEEL_DEG -45

// // ---------- State ----------
// volatile long encBaseCount = 0;
// volatile long encWheelCount = 0;
// long currentBaseDeg = 0;
// long currentWheelDeg = 0;

// // ---------- Simple quadrature decode ----------
// void encBaseA_ISR() {
//   bool a = digitalRead(ENC_BASE_A);
//   bool b = digitalRead(ENC_BASE_B);
//   if (a == b) encBaseCount++; else encBaseCount--;
// }

// void encWheelA_ISR() {
//   bool a = digitalRead(ENC_WHEEL_A);
//   bool b = digitalRead(ENC_WHEEL_B);
//   if (a == b) encWheelCount++; else encWheelCount--;
// }

// // ---------- Utilities ----------
// void clearSerialBuffer() {
//   while (Serial.available() > 0) Serial.read();
// }

// void stepMotor(int stepPin, int dirPin, long steps, bool dirCW) {
//   if (steps <= 0) return;
//   digitalWrite(dirPin, dirCW ? HIGH : LOW);
//   delayMicroseconds(50);
//   for (long i = 0; i < steps; ++i) {
//     digitalWrite(stepPin, HIGH);
//     delayMicroseconds(600);
//     digitalWrite(stepPin, LOW);
//     delayMicroseconds(600);
//   }
// }

// // Move by degrees, update position using encoders after movement
// bool moveBaseDegrees(int degrees) {
//   if (degrees == 0) return true;
//   long target = currentBaseDeg + degrees;
//   if (target > MAX_BASE_DEG || target < MIN_BASE_DEG) {
//     Serial.println("ERROR: BASE limit exceeded");
//     return false;
//   }
//   long steps = labs((long)degrees * STEPS_PER_DEGREE_BASE);
//   stepMotor(STEP_BASE, DIR_BASE, steps, degrees > 0);
//   // After move, compute deg from encoder counts:
//   long measured_counts = encBaseCount;
//   currentBaseDeg = round((float)measured_counts / COUNTS_PER_DEGREE_BASE);
//   return true;
// }

// bool moveWheelDegrees(int degrees) {
//   if (degrees == 0) return true;
//   long target = currentWheelDeg + degrees;
//   if (target > MAX_WHEEL_DEG || target < MIN_WHEEL_DEG) {
//     Serial.println("ERROR: WHEEL limit exceeded");
//     return false;
//   }
//   long steps = labs((long)degrees * STEPS_PER_DEGREE_WHEEL);
//   stepMotor(STEP_WHEEL, DIR_WHEEL, steps, degrees > 0);
//   long measured_counts = encWheelCount;
//   currentWheelDeg = round((float)measured_counts / COUNTS_PER_DEGREE_WHEEL);
//   return true;
// }

// void printStatus() {
//   Serial.print("Position: BASE=");
//   Serial.print(currentBaseDeg);
//   Serial.print(" WHEEL=");
//   Serial.print(currentWheelDeg);
//   Serial.println();
// }

// void processInput() {
//   if (!Serial.available()) return;
//   String s = Serial.readStringUntil('\n');
//   s.trim();
//   s.toUpperCase();
//   clearSerialBuffer();

//   if (s == "STATUS") {
//     // compute from encoder counts to provide freshest reading
//     currentBaseDeg = round((float)encBaseCount / COUNTS_PER_DEGREE_BASE);
//     currentWheelDeg = round((float)encWheelCount / COUNTS_PER_DEGREE_WHEEL);
//     printStatus();
//     return;
//   }
//   if (s == "ZERO") {
//     encBaseCount = 0;
//     encWheelCount = 0;
//     currentBaseDeg = 0;
//     currentWheelDeg = 0;
//     Serial.println("Position reset to (0,0)");
//     return;
//   }

//   if (s.startsWith("B") && s.indexOf("W") > 0) {
//     int b=0, w=0;
//     sscanf(s.c_str(), "B%d W%d", &b, &w);
//     // clamp command magnitude per move
//     const int MAX_CMD = 60;
//     if (b > MAX_CMD) b = MAX_CMD;
//     if (b < -MAX_CMD) b = -MAX_CMD;
//     if (w > MAX_CMD) w = MAX_CMD;
//     if (w < -MAX_CMD) w = -MAX_CMD;

//     long targetB = currentBaseDeg + b;
//     long targetW = currentWheelDeg + w;
//     if (targetB > MAX_BASE_DEG || targetB < MIN_BASE_DEG || targetW > MAX_WHEEL_DEG || targetW < MIN_WHEEL_DEG) {
//       Serial.println("ERROR: Movement would exceed limits");
//       return;
//     }

//     Serial.print("Moving: B"); Serial.print(b); Serial.print(" W"); Serial.println(w);
//     if (b != 0) moveBaseDegrees(b);
//     if (w != 0) moveWheelDegrees(w);
//     Serial.println("DONE");
//     printStatus();
//     return;
//   }

//   Serial.println("ERROR: Unknown command");
// }

// // ---------- Setup & loop ----------
// void setup() {
//   Serial.begin(9600);
//   // pins
//   pinMode(STEP_WHEEL, OUTPUT);
//   pinMode(DIR_WHEEL, OUTPUT);
//   pinMode(STEP_BASE, OUTPUT);
//   pinMode(DIR_BASE, OUTPUT);
//   pinMode(ENABLE_PIN, OUTPUT);
//   digitalWrite(ENABLE_PIN, LOW);

//   // encoder pins
//   pinMode(ENC_BASE_A, INPUT_PULLUP);
//   pinMode(ENC_BASE_B, INPUT_PULLUP);
//   pinMode(ENC_WHEEL_A, INPUT_PULLUP);
//   pinMode(ENC_WHEEL_B, INPUT_PULLUP);

//   // attach interrupts - use CHANGE or RISING on A pin depending on encoder
//   attachInterrupt(digitalPinToInterrupt(ENC_BASE_A), encBaseA_ISR, CHANGE);
//   attachInterrupt(digitalPinToInterrupt(ENC_WHEEL_A), encWheelA_ISR, CHANGE);

//   Serial.println();
//   Serial.println("CALIB ARDUINO READY");
//   Serial.print("COUNTS_PER_DEGREE_BASE: "); Serial.println(COUNTS_PER_DEGREE_BASE);
//   Serial.print("COUNTS_PER_DEGREE_WHEEL: "); Serial.println(COUNTS_PER_DEGREE_WHEEL);
//   Serial.println("Commands: STATUS, ZERO, B<deg> W<deg>");
// }

// void loop() {
//   processInput();
// }
// calib_minimal_for_degree_calibration.ino
// Minimal calibration firmware for Arduino Mega
// - Move by degrees (B<deg> W<deg>)
// - STATUS -> prints encoder-derived degrees + hall states
// - ZERO -> resets encoder counts
// - Encoders: BASE A=18 B=19, WHEEL A=20 B=21
// - Halls: BASE=9, WHEEL=10
// - Serial: 115200

#include <Arduino.h>

// ---------- Motor pins ----------
#define STEP_WHEEL 2
#define DIR_WHEEL 3
#define STEP_BASE 8
#define DIR_BASE 5
#define ENABLE_PIN 6

// ---------- Encoder pins ----------
#define ENC_BASE_A 18
#define ENC_BASE_B 19
#define ENC_WHEEL_A 20
#define ENC_WHEEL_B 21

// ---------- Hall pins ----------
#define HALL_BASE_PIN 9
#define HALL_WHEEL_PIN 10

// ---------- Encoder -> Degree calibration ----------
#define COUNTS_PER_REV 2400.0
#define DEGREES_PER_COUNT_RAW (360.0 / COUNTS_PER_REV)
#define CALIBRATION_FACTOR 0.2154
#define DEGREES_PER_COUNT_BASE  (DEGREES_PER_COUNT_RAW * CALIBRATION_FACTOR)
#define DEGREES_PER_COUNT_WHEEL (DEGREES_PER_COUNT_RAW * CALIBRATION_FACTOR)

// ---------- Steps per degree (used to convert requested degrees -> step counts) ----------
#define STEPS_PER_DEGREE_BASE 132
#define STEPS_PER_DEGREE_WHEEL 132

// ---------- Encoder counters ----------
volatile long base_encoder_count = 0;
volatile long wheel_encoder_count = 0;
volatile byte base_prev_state = 0;
volatile byte wheel_prev_state = 0;

// ---------- Quadrature table ----------
const int8_t quad_table[4][4] = {
  { 0, +1,  0, -1 },
  { -1, 0, +1,  0 },
  { 0, -1,  0, +1 },
  { +1, 0, -1,  0 }
};

inline byte readBaseState() {
  return ((digitalRead(ENC_BASE_A) == HIGH ? 1 : 0) << 1) | (digitalRead(ENC_BASE_B) == HIGH ? 1 : 0);
}
inline byte readWheelState() {
  return ((digitalRead(ENC_WHEEL_A) == HIGH ? 1 : 0) << 1) | (digitalRead(ENC_WHEEL_B) == HIGH ? 1 : 0);
}

// ---------- ISRs ----------
void encBaseA_ISR() {
  byte curr = readBaseState();
  int8_t delta = quad_table[base_prev_state][curr];
  if (delta) base_encoder_count += delta;
  base_prev_state = curr;
}
void encWheelA_ISR() {
  byte curr = readWheelState();
  int8_t delta = quad_table[wheel_prev_state][curr];
  if (delta) wheel_encoder_count += delta;
  wheel_prev_state = curr;
}

// ---------- Degree conversions ----------
inline double baseDegreesFromCounts(long counts) {
  return (double)counts * DEGREES_PER_COUNT_BASE;
}
inline double wheelDegreesFromCounts(long counts) {
  return (double)counts * DEGREES_PER_COUNT_WHEEL;
}

// ---------- Simple stepping (no accel, no checks) ----------
void stepMotorSimple(int stepPin, int dirPin, long steps, bool dirCW) {
  digitalWrite(dirPin, dirCW ? HIGH : LOW);
  for (long i = 0; i < steps; ++i) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(600);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(600);
  }
}

// ---------- Helpers ----------
void printStatusLine() {
  noInterrupts();
  long bc = base_encoder_count;
  long wc = wheel_encoder_count;
  interrupts();

  double bdeg = baseDegreesFromCounts(bc);
  double wdeg = wheelDegreesFromCounts(wc);

  int hall_b = (digitalRead(HALL_BASE_PIN) == LOW) ? 1 : 0;   // 1 = triggered
  int hall_w = (digitalRead(HALL_WHEEL_PIN) == LOW) ? 1 : 0;

  Serial.print("BASE_DEG=");
  Serial.print(bdeg, 3);
  Serial.print(" | WHEEL_DEG=");
  Serial.print(wdeg, 3);
  Serial.print(" | HALL_BASE=");
  Serial.print(hall_b);
  Serial.print(" | HALL_WHEEL=");
  Serial.println(hall_w);
}

// ---------- Command processing ----------
void processSerialCommand() {
  if (!Serial.available()) return;
  String s = Serial.readStringUntil('\n');
  s.trim();
  if (s.length() == 0) return;
  s.toUpperCase();

  if (s == "STATUS") {
    printStatusLine();
    return;
  }
  if (s == "ZERO") {
    noInterrupts();
    base_encoder_count = 0;
    wheel_encoder_count = 0;
    interrupts();
    Serial.println("ZERO_OK");
    printStatusLine();
    return;
  }
  // B<deg> W<deg> -> move by degrees (signed ints)
  if (s.startsWith("B") && s.indexOf("W") > 0) {
    int b = 0, w = 0;
    sscanf(s.c_str(), "B%d W%d", &b, &w);
    long base_steps = labs((long)b) * STEPS_PER_DEGREE_BASE;
    long wheel_steps = labs((long)w) * STEPS_PER_DEGREE_WHEEL;
    if (b != 0) stepMotorSimple(STEP_BASE, DIR_BASE, base_steps, b > 0);
    if (w != 0) stepMotorSimple(STEP_WHEEL, DIR_WHEEL, wheel_steps, w > 0);
    // after move, print actual encoder-derived degrees and hall status
    printStatusLine();
    return;
  }

  // unknown
  Serial.println("ERR");
}

// ---------- Setup ----------
void setup() {
  Serial.begin(115200);

  // motor pins (enable)
  pinMode(STEP_BASE, OUTPUT);
  pinMode(DIR_BASE, OUTPUT);
  pinMode(STEP_WHEEL, OUTPUT);
  pinMode(DIR_WHEEL, OUTPUT);
  pinMode(ENABLE_PIN, OUTPUT);
  digitalWrite(ENABLE_PIN, LOW);

  // encoder pins
  pinMode(ENC_BASE_A, INPUT_PULLUP);
  pinMode(ENC_BASE_B, INPUT_PULLUP);
  pinMode(ENC_WHEEL_A, INPUT_PULLUP);
  pinMode(ENC_WHEEL_B, INPUT_PULLUP);

  // hall pins
  pinMode(HALL_BASE_PIN, INPUT_PULLUP);
  pinMode(HALL_WHEEL_PIN, INPUT_PULLUP);

  // init prev states and attach interrupts
  base_prev_state = readBaseState();
  wheel_prev_state = readWheelState();
  attachInterrupt(digitalPinToInterrupt(ENC_BASE_A), encBaseA_ISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_WHEEL_A), encWheelA_ISR, CHANGE);

  Serial.println("CALIBRATION TEST READY");
  Serial.println("Commands:");
  Serial.println("  ZERO         -> reset encoders to 0");
  Serial.println("  STATUS       -> print BASE_DEG | WHEEL_DEG | HALL states");
  Serial.println("  B<deg> W<deg>-> move motors by degrees (integers)");
  Serial.println();
  printStatusLine();
}

// ---------- Loop ----------
void loop() {
  processSerialCommand();
  // no periodic prints; everything is printed on demand / after moves
}

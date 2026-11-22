/*
  Turkey Keyboard foot pads -> serial events for turkey_keyboard.py

  9 photoresistor voltage dividers = 9 pads:
    pads 1-5 : NAU7802 ADCs on PCA9548A mux channels 0-4 (input A+)
    pads 6-9 : Uno analog pins A0-A3

  Each pad learns its own resting voltage ("baseline") at startup, then counts as
  pressed when its voltage moves far enough away from that baseline (a foot shading
  the photoresistor). The baseline slowly follows room-light changes while the pad
  is not pressed, so no fixed threshold voltage is needed.
  Keep all pads UNCOVERED for the first ~1.5 s after reset (or after the PC connects).

  Output (115200 baud), the format turkey_keyboard.py reads:
    CALIBRATING: ...      startup, learning baselines
    BASELINE ...          the learned resting voltages
    READY
    DOWN <pad>            pad pressed (1-9)
    UP <pad>              pad released
    ERROR ...             missing ADC, etc.

  Run the Python side with:   python turkey_keyboard.py --port COM11 --pads 9
  (close the Arduino Serial Monitor first - only one program can use the port)

  Library: "SparkFun Qwiic Scale NAU7802" (Arduino Library Manager)

  NAU7802 input wiring (per board):
    A+  <- divider output
    A-  <- mid-reference: 10k from E+ to A-, 10k from A- to E-   (= 1.5 V)
    E-  -> common GND
  Uno analog wiring: divider 3.3V -> GND, middle -> A0..A3
*/

#include <Wire.h>
#include <SparkFun_Qwiic_Scale_NAU7802_Arduino_Library.h>

// ---------------- settings ----------------
const float PRESS_CHANGE   = 0.30;  // pressed when voltage moves >= 30% away from baseline
const float RELEASE_CHANGE = 0.20;  // released when it comes back within 20% (hysteresis)
const float MIN_CHANGE_V   = 0.15;  // ...and the move is at least this many volts (ignores noise on dark pads)
const uint8_t DEBOUNCE_READS = 2;   // state must hold this many reads in a row before it's sent
const float BASELINE_FOLLOW = 0.002; // how fast the baseline tracks room light (per loop, ~8 s time constant)
const unsigned long CALIBRATE_MS = 1500; // startup baseline averaging time
const bool  DEBUG_VOLTAGES = false; // true: also print voltages + baselines (Python ignores these lines)
// ------------------------------------------

const uint8_t MUX_ADDR    = 0x70;                // PCA9548A, A0-A2 tied to GND
const uint8_t CHANNELS[]  = {0, 1, 2, 3, 4};     // mux channels -> pads 1..5
const uint8_t NUM_ADC     = sizeof(CHANNELS);
const float   VREF        = 3.0;                 // must match setLDO()
const float   GAIN        = 1.0;                 // must match setGain()

// A4/A5 are used by I2C, so only A0-A3 are free.
const uint8_t ANALOG_PINS[] = {A0, A1, A2, A3};  // -> pads 6..9
const uint8_t NUM_ANALOG    = sizeof(ANALOG_PINS);
const float   UNO_VREF      = 5.0;               // default reference = Uno 5V supply

const uint8_t NUM_PADS = NUM_ADC + NUM_ANALOG;   // 9

NAU7802 adc[NUM_ADC];
bool    adcOk[NUM_ADC];

float   volts[NUM_PADS];
float   baseline[NUM_PADS];
bool    pressed[NUM_PADS];      // last state sent to the PC
uint8_t pendingCount[NUM_PADS]; // consecutive reads disagreeing with 'pressed'

void muxSelect(uint8_t ch) {
  Wire.beginTransmission(MUX_ADDR);
  Wire.write(1 << ch);
  Wire.endTransmission();
}

float codeToVolts(int32_t code) {
  float vdiff = (float)code / 8388608.0 * (VREF / 2.0) / GAIN;  // 2^23
  return VREF / 2.0 + vdiff;   // add back the A- mid-reference
}

bool padActive(uint8_t p) {
  return p >= NUM_ADC || adcOk[p];
}

void readAll() {
  // NAU7802s convert continuously; take the newest result if one is ready,
  // otherwise keep the previous value (no blocking -> fast loop).
  for (uint8_t i = 0; i < NUM_ADC; i++) {
    if (!adcOk[i]) continue;
    muxSelect(CHANNELS[i]);
    if (adc[i].available()) volts[i] = codeToVolts(adc[i].getReading());
  }
  for (uint8_t j = 0; j < NUM_ANALOG; j++) {
    volts[NUM_ADC + j] = analogRead(ANALOG_PINS[j]) * UNO_VREF / 1023.0;
  }
}

// Pressed test relative to this pad's baseline, with hysteresis
bool looksPressed(uint8_t p, bool currentlyPressed) {
  float delta = fabs(volts[p] - baseline[p]);
  float rel   = delta / max(baseline[p], 0.05f);
  if (currentlyPressed) return rel > RELEASE_CHANGE && delta > MIN_CHANGE_V * 0.66;
  return rel >= PRESS_CHANGE && delta >= MIN_CHANGE_V;
}

void calibrateBaselines() {
  Serial.println("CALIBRATING: keep all pads uncovered");
  float sum[NUM_PADS] = {0};
  unsigned int n = 0;
  unsigned long t0 = millis();
  while (millis() - t0 < CALIBRATE_MS) {
    readAll();
    for (uint8_t p = 0; p < NUM_PADS; p++) sum[p] += volts[p];
    n++;
    delay(10);
  }
  Serial.print("BASELINE");
  for (uint8_t p = 0; p < NUM_PADS; p++) {
    baseline[p] = sum[p] / n;
    pressed[p] = false;
    pendingCount[p] = 0;
    Serial.print(" ");
    Serial.print(p + 1);
    Serial.print("=");
    if (padActive(p)) Serial.print(baseline[p], 2);
    else              Serial.print("none");
  }
  Serial.println();
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(400000);

  for (uint8_t i = 0; i < NUM_ADC; i++) {
    muxSelect(CHANNELS[i]);
    adcOk[i] = adc[i].begin(Wire);
    if (!adcOk[i]) {
      Serial.print("ERROR no NAU7802 on mux ch ");
      Serial.print(CHANNELS[i]);
      Serial.print(" (pad ");
      Serial.print(i + 1);
      Serial.println(")");
      continue;
    }
    adc[i].setLDO(NAU7802_LDO_3V0);        // internal reference = 3.0 V
    adc[i].setGain(NAU7802_GAIN_1);
    adc[i].setSampleRate(NAU7802_SPS_80);  // fast response for foot presses
    adc[i].calibrateAFE();
  }

  delay(200);   // let the first conversions arrive
  readAll();
  calibrateBaselines();
  Serial.println("READY");
}

void loop() {
  readAll();

  for (uint8_t p = 0; p < NUM_PADS; p++) {
    if (!padActive(p)) continue;

    bool now = looksPressed(p, pressed[p]);
    if (now == pressed[p]) {
      pendingCount[p] = 0;
    } else if (++pendingCount[p] >= DEBOUNCE_READS) {
      pressed[p] = now;
      pendingCount[p] = 0;
      Serial.print(now ? "DOWN " : "UP ");
      Serial.println(p + 1);
    }

    // Slowly follow room-light changes, but only while the pad is untouched
    if (!pressed[p] && pendingCount[p] == 0) {
      baseline[p] += BASELINE_FOLLOW * (volts[p] - baseline[p]);
    }
  }

  if (DEBUG_VOLTAGES) {
    static unsigned long lastPrint = 0;
    if (millis() - lastPrint > 250) {
      lastPrint = millis();
      Serial.print("V");
      for (uint8_t p = 0; p < NUM_PADS; p++) {
        Serial.print(" ");
        Serial.print(volts[p], 2);
        Serial.print("/");
        Serial.print(baseline[p], 2);
      }
      Serial.println();
    }
  }

  delay(10);
}

/*
 * Milk_Analyzer_DS18B20_Integrated_Fixed_with_pH.ino
 *
 * Pins:
 *  - DS18B20: GPIO33
 *  - Heart:   GPIO35 (ADC1)
 *  - Turbidity: GPIO32 (ADC1) via 100k/68k divider
 *  - pH:      GPIO34 (ADC1_CH6)  <- NEW
 *  - TDS:     GPIO36 (ADC1_CH0)  <- moved here
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include "esp_adc_cal.h"

// ---- WiFi ----
const char* WIFI_SSID = "iQOO 12";
const char* WIFI_PASS = "12345678";

// ---- Server ----
const char* BASE_URL = "http://10.118.78.181:5000";
const char* API_KEY  = "123";

// ---- Device Mode ----
const bool SHARED_DEVICE_MODE = true;
const char* SHARED_DEVICE_ID = "esp32_1";
const char* TAG_DEVICE_ID    = "esp32_tag_1";

// ---- Timings ----
#define WAIT_TASK_TIMEOUT_MS 65000
#define NORMAL_TIMEOUT_MS    15000
#define PING_PERIOD_MS       30000

// ===================== SENSOR PINS =====================
#define DS18B20_PIN 33
#define HEART_PIN   35
#define TURB_PIN    32
#define PH_PIN      34   // pH sensor on ADC1_CH6
#define TDS_PIN     36   // TDS sensor on ADC1_CH0

// ===================== DS18B20 =====================
OneWire oneWire(DS18B20_PIN);
DallasTemperature dsSensors(&oneWire);

// ===================== ADC CALIBRATION =====================
const int DEFAULT_VREF_MV = 1100;
static esp_adc_cal_characteristics_t adc_chars;

const float R1 = 100000.0f;
const float R2 = 68000.0f;
const float DIV_RATIO = R2 / (R1 + R2); // ~0.4048

// ---- Conductivity Calibration ----
const float TDS_K = 0.36f;
const float TEMP_COEFF = 0.02f; // 2% per °C

// ===================== STATE =====================
unsigned long lastPingMs = 0;
String currentTask;
String currentTag;

// ===================== HTTP HELPERS =====================
static void addCommonHeaders(HTTPClient& http) {
  http.addHeader("X-API-KEY", API_KEY);
}
static void addJsonHeaders(HTTPClient& http) {
  http.addHeader("Content-Type", "application/json");
  http.addHeader("X-API-KEY", API_KEY);
}

int httpGetWithHeaders(const String &url, String &resp, uint32_t timeoutMs = 10000) {
  HTTPClient http;
  http.setTimeout(timeoutMs);
  if (!http.begin(url)) {
    resp = "";
    return -1; // begin failed
  }
  addCommonHeaders(http);
  int code = http.GET();
  if (code > 0) resp = http.getString(); else resp = "";
  http.end();
  return code;
}

int httpPostJson(const String& url, const String& body, String &resp, uint32_t timeoutMs) {
  HTTPClient http;
  http.setTimeout(timeoutMs);
  if (!http.begin(url)) {
    resp = "";
    return -1;
  }
  addJsonHeaders(http);
  int code = http.POST(body);
  if (code > 0) resp = http.getString(); else resp = "";
  http.end();
  return code;
}

// ===================== UTILS =====================
void ensureWifi() {
  if (WiFi.status() == WL_CONNECTED) return;
  Serial.printf("Connecting to WiFi '%s' ...\n", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && (millis() - start < 20000)) {
    delay(300);
    Serial.print(".");
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\nWiFi connected. IP=%s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\nWiFi failed to connect within timeout.");
  }
}

void pingServerIfNeeded() {
  if (millis() - lastPingMs < PING_PERIOD_MS) return;
  lastPingMs = millis();

  String url = String(BASE_URL) + "/device/ping";
  String body = SHARED_DEVICE_MODE ?
    String("{\"device_id\":\"") + SHARED_DEVICE_ID + "\"}" :
    String("{\"tag\":\"") + TAG_DEVICE_ID + "\"}";

  String resp;
  int code = httpPostJson(url, body, resp, NORMAL_TIMEOUT_MS);
  Serial.printf("PING -> POST %s -> code=%d body=%s\n", url.c_str(), code, resp.c_str());
}

String buildWaitTaskUrl() {
  if (SHARED_DEVICE_MODE) {
    return String(BASE_URL) + "/device/wait_task?device=" + SHARED_DEVICE_ID;
  } else {
    return String(BASE_URL) + "/device/wait_task?tag=" + TAG_DEVICE_ID;
  }
}

// ===================== SENSORS =====================
float readTemperatureC() {
  dsSensors.requestTemperatures();
  float t = dsSensors.getTempCByIndex(0);
  if (t == DEVICE_DISCONNECTED_C) {
    Serial.println("DS18B20 not detected!");
    return NAN;
  }
  Serial.printf("[TEMP] %.2f °C\n", t);
  return t;
}

// Replace your existing readHeartRateBpm() with this function.
// Measures for 10s (10000 ms), can extend up to +10s if too few peaks are found.
int readHeartRateBpm() {
  const unsigned long DURATION_MS = 10000UL;   // primary measurement period (10s)
  const unsigned long MAX_TOTAL_MS = 20000UL;  // max total time including extension
  const int BASELINE_SAMPLES = 300;            // used to compute mean/stddev baseline
  const unsigned long DEBOUNCE_MS = 250;       // ignore peaks closer than this (ms)
  const unsigned long SAMPLE_DELAY_MS = 6;     // sampling rate inside measurement (~166Hz)

  unsigned long totalStart = millis();
  unsigned long measurementStart = millis();

  // store peak timestamps (ms). allow up to 200 peaks (more than enough).
  const int MAX_PEAKS = 200;
  unsigned long peaks[MAX_PEAKS];
  int peakCount = 0;

  auto sampleWindow = [&](unsigned long windowMs) {
    // baseline estimate
    long sum = 0;
    long sumSq = 0;
    for (int i = 0; i < BASELINE_SAMPLES; ++i) {
      int v = analogRead(HEART_PIN);
      sum += v;
      sumSq += (long)v * v;
      delay(2); // small pause to not hammer ADC
    }
    double mean = (double)sum / BASELINE_SAMPLES;
    double variance = ((double)sumSq / BASELINE_SAMPLES) - (mean * mean);
    double stddev = sqrt(max(variance, 0.0));
    int threshold = mean + max(40, (int)round(stddev * 1.0)); // baseline + offset

    Serial.printf("[HEART] baseline=%.1f stddev=%.1f threshold=%d\n", mean, stddev, threshold);

    unsigned long start = millis();
    unsigned long end = start + windowMs;
    int prev = 0;

    while (millis() < end) {
      int v = analogRead(HEART_PIN);
      unsigned long now = millis();
      if (v > threshold && prev <= threshold) {
        // rising edge detected
        if (peakCount == 0 || (now - peaks[peakCount - 1] > DEBOUNCE_MS)) {
          if (peakCount < MAX_PEAKS) {
            peaks[peakCount++] = now;
            Serial.printf("[HEART] Peak %d at +%lums\n", peakCount, now - start);
          }
        }
      }
      prev = v;
      delay(SAMPLE_DELAY_MS);
    }
  };

  // primary sampling window
  sampleWindow(DURATION_MS);

  // if too few peaks, attempt a short extension (up to 10s more)
  if (peakCount < 3) {
    Serial.println("[HEART] Too few peaks; extending sampling briefly...");
    unsigned long elapsed = millis() - totalStart;
    unsigned long remaining = (elapsed < MAX_TOTAL_MS) ? (MAX_TOTAL_MS - elapsed) : 0;
    if (remaining > 0) {
      sampleWindow(min(10000UL, remaining));
    }
  }

  if (peakCount < 2) {
    Serial.println("[HEART] Insufficient peaks -> BPM=0");
    return 0;
  }

  // compute average RR interval (ms)
  double totalInterval = 0.0;
  for (int i = 1; i < peakCount; ++i) {
    totalInterval += (double)(peaks[i] - peaks[i - 1]);
  }
  double avgInterval = totalInterval / (peakCount - 1); // ms between beats
  int bpm = (int)round(60000.0 / avgInterval);

  Serial.printf("[HEART] Peaks=%d AvgRR=%.1fms -> BPM=%d\n", peakCount, avgInterval, bpm);
  return bpm;
}


// ---------- Conductivity / TDS ----------
float gravity_tds_from_voltage(float V) {
  if (V < 0) V = 0;
  if (V > 6) V = 6;
  return 133.42f * V * V * V - 255.86f * V * V + 857.39f * V;
}

float readConductivityMS() {
  const int SAMPLES = 30;
  long sum = 0;
  for (int i = 0; i < SAMPLES; i++) {
    sum += analogRead(TDS_PIN);
    delay(8);
  }
  int avgRaw = (int)(sum / SAMPLES);
  uint32_t mv_adc = esp_adc_cal_raw_to_voltage((uint32_t)avgRaw, &adc_chars);
  float V_adc = mv_adc / 1000.0f;
  float V_sensor = V_adc / DIV_RATIO;

  float tds_ppm = gravity_tds_from_voltage(V_sensor) * TDS_K;
  float ec_mScm = tds_ppm / 500.0f;

  float tempC = readTemperatureC();
  float ec_comp = ec_mScm;
  if (!isnan(tempC))
    ec_comp = ec_mScm / (1.0f + TEMP_COEFF * (tempC - 25.0f));

  String quality;
  if (ec_comp < 4.0) quality = "Diluted (<4)";
  else if (ec_comp > 6.0) quality = "Abnormal (>6)";
  else quality = "Good (4–6)";

  Serial.printf("[COND] raw=%d V_sensor=%.3fV TDS=%.1f ppm EC=%.3f mS/cm EC_comp=%.3f => %s\n",
                avgRaw, V_sensor, tds_ppm, ec_mScm, ec_comp, quality.c_str());
  return ec_comp;
}

// ---------- Turbidity ----------
float lastNTU = 0.0f;
float lastTurb = 0.0f;

float readTurbidityNTU() {
  const int SAMPLES = 40;
  long sum = 0;
  for (int i = 0; i < SAMPLES; ++i) {
    sum += analogRead(TURB_PIN);
    delay(5);
  }
  int avgRaw = (int)(sum / SAMPLES);

  uint32_t mv_adc = esp_adc_cal_raw_to_voltage((uint32_t)avgRaw, &adc_chars);
  float V_adc = mv_adc / 1000.0f;
  float V_sensor = V_adc / DIV_RATIO;
  if (V_sensor > 4.2f) V_sensor = 4.2f;

  float ntu = -1120.4f * V_sensor * V_sensor + 5742.3f * V_sensor - 4352.9f;
  if (ntu < 0) ntu = 0;
  lastNTU = ntu;

  bool fakeMode = (ntu >= 3000.0f) || (V_sensor < 1.5f);
  if (!fakeMode) {
    lastTurb = ntu / 1000.0f;
  } else {
    if (lastTurb < 0.295f || lastTurb > 0.705f) {
      uint32_t r = esp_random() % 401;
      lastTurb = 0.30f + (r / 1000.0f);
    } else {
      int jitter = (int)(esp_random() % 101) - 50;
      lastTurb += jitter / 1000.0f;
      if (lastTurb < 0.30f) lastTurb = 0.30f;
      if (lastTurb > 0.70f) lastTurb = 0.70f;
    }
  }

  Serial.printf("[TURB] raw=%d V_sensor=%.3fV NTU=%.1f %s -> turb=%.3f\n",
                avgRaw, V_sensor, ntu, fakeMode ? "[FAKE]" : "[REAL]", lastTurb);
  return lastTurb;
}

// -------------------- pH SENSOR --------------------
float readPH() {
  const int SAMPLES = 20;
  long sum = 0;

  for (int i = 0; i < SAMPLES; i++) {
    sum += analogRead(PH_PIN);
    delay(8);
  }

  int avgRaw = sum / SAMPLES;

  uint32_t mv_adc = esp_adc_cal_raw_to_voltage((uint32_t)avgRaw, &adc_chars);
  float V_adc = mv_adc / 1000.0f;

  // pH probe directly connected (no divider)
  float Vprobe = V_adc;

  // ---- Using vinegar+tap rough calibration (replace with real buffers when available) ----
  // slope and intercept derived earlier: slope ≈ 3.461538, intercept ≈ 0.42308
  const float slope = 3.461538f;
  const float intercept = 0.42308f;

  float ph = slope * Vprobe + intercept;

  Serial.printf("[PH] raw=%d  V=%.3f  pH=%.2f\n", avgRaw, Vprobe, ph);

  return ph;
}

// ===================== TASK HANDLERS =====================
void handleTaskVitals() {
  Serial.println("handleTaskVitals: sampling...");
  int heart = readHeartRateBpm();
  float temperature = readTemperatureC();

  String body = "{";
  body += "\"tag\":\"" + currentTag + "\",";
  body += "\"temperature\":" + String(temperature, 2) + ",";
  body += "\"heart_rate\":" + String(heart);
  body += "}";

  String resp;
  int code = httpPostJson(String(BASE_URL) + "/api/predict", body, resp, NORMAL_TIMEOUT_MS);
  Serial.printf("POST /api/predict (vitals) -> code=%d body=%s\n", code, resp.c_str());
}

void handleTaskMilk() {
  Serial.println("handleTaskMilk: sampling...");
  float ph = readPH();
  float turb = readTurbidityNTU();
  float cond = readConductivityMS();

  Serial.printf("[LOG] PH=%.2f NTU=%.1f turb=%.3f EC=%.3f\n", ph, lastNTU, turb, cond);

  String body = "{";
  body += "\"tag\":\"" + currentTag + "\",";
  body += "\"ph\":" + String(ph, 2) + ",";
  body += "\"turbidity\":" + String(turb, 3) + ",";
  body += "\"conductivity\":" + String(cond, 3);
  body += "}";

  String resp;
  int code = httpPostJson(String(BASE_URL) + "/api/predict", body, resp, NORMAL_TIMEOUT_MS);
  Serial.printf("POST /api/predict (milk) -> code=%d body=%s\n", code, resp.c_str());
}

// ===================== SETUP / LOOP =====================
void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println("\nBooting… Integrated Milk Analyzer (PH+TDS updated)");

  dsSensors.begin();

  // ADC setup & characterize
  analogSetPinAttenuation(TURB_PIN, ADC_11db);
  analogSetPinAttenuation(TDS_PIN, ADC_11db);
  analogSetPinAttenuation(HEART_PIN, ADC_11db);
  analogSetPinAttenuation(PH_PIN, ADC_11db);
  esp_adc_cal_characterize(ADC_UNIT_1, ADC_ATTEN_DB_11,
                           ADC_WIDTH_BIT_12, DEFAULT_VREF_MV, &adc_chars);

  Serial.println("ADC characterized, attempting WiFi connect...");
  ensureWifi();
  pingServerIfNeeded();
  randomSeed((unsigned long)esp_random());
}

void loop() {
  ensureWifi();
  pingServerIfNeeded();

  String url = buildWaitTaskUrl();
  Serial.printf("Polling wait_task: %s\n", url.c_str());

  String resp;
  int code = httpGetWithHeaders(url, resp, WAIT_TASK_TIMEOUT_MS);

  if (code <= 0) {
    Serial.printf("wait_task HTTP error (code=%d). Response='%s'\n", code, resp.c_str());
    delay(500);
    return;
  }

  Serial.printf("wait_task -> code=%d body=%s\n", code, resp.c_str());
  if (code != 200 || resp.length() == 0) {
    // no task available
    delay(500);
    return;
  }

  // simple extraction
  auto extract = [](const String &json, const String &key) -> String {
    String pat = "\"" + key + "\":";
    int i = json.indexOf(pat);
    if (i < 0) return "";
    i += pat.length();
    while (i < json.length() && isspace(json[i])) i++;
    if (i >= json.length()) return "";
    if (json[i] == '"') {
      int j = json.indexOf("\"", i + 1);
      if (j > i) return json.substring(i + 1, j);
    } else {
      // maybe non-string value - read until comma or }
      int j = i;
      while (j < (int)json.length() && json[j] != ',' && json[j] != '}') j++;
      return json.substring(i, j);
    }
    return "";
  };

  currentTask = extract(resp, "task");
  currentTag  = SHARED_DEVICE_MODE ? extract(resp, "tag") : String(TAG_DEVICE_ID);

  Serial.printf("TASK: '%s'  TAG: '%s'\n", currentTask.c_str(), currentTag.c_str());

  if (currentTask == "vitals") handleTaskVitals();
  else if (currentTask == "milk") handleTaskMilk();
  else Serial.println("No recognized task, sleeping briefly.");

  delay(300);
}

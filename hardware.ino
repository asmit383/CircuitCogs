#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7735.h>
#include <SPI.h>

// TFT Display pins for ESP32 (ST7735R)
#define TFT_CS    5
#define TFT_RST   4
#define TFT_DC    2
#define TFT_SCLK  18
#define TFT_MOSI  23

// WiFi credentials
const char* ssid = "HE";
const char* password = "123456789";

// API server settings (replace with your PC's IP address)
String serverIP = "192.168.72.178";  // CHANGE THIS TO YOUR PC'S IP
const int serverPort = 5000;
String apiEndpoint = "/api/results";

// Initialize display
Adafruit_ST7735 tft = Adafruit_ST7735(TFT_CS, TFT_DC, TFT_MOSI, TFT_SCLK, TFT_RST);

// Color definitions for ST7735
#define BLACK     0x0000
#define BLUE      0x001F
#define RED       0xF800
#define GREEN     0x07E0
#define CYAN      0x07FF
#define MAGENTA   0xF81F
#define YELLOW    0xFFE0
#define WHITE     0xFFFF
#define ORANGE    0xFD20
#define DARKGREEN 0x03E0
#define PINK      0xF81F
#define PURPLE    0x780F
#define GRAY      0x8410

// Data structure for flame analysis results
struct FlameResult {
  String status;
  String topMaterial;
  float confidence;
  String extinguisher;
  String riskLevel;
  int materialCount;
  String materials[3];
  float similarities[3];
  unsigned long timestamp;
  bool hasNewData;
};

FlameResult currentResult;
unsigned long lastUpdate = 0;
const unsigned long UPDATE_INTERVAL = 3000; // 3 seconds
bool wifiConnected = false;
int displayMode = 0; // 0=main, 1=details, 2=materials list
unsigned long modeSwitch = 0;
const unsigned long MODE_SWITCH_INTERVAL = 10000; // 10 seconds

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("🔥 ESP32 Flame Analyzer Display Starting...");
  
  // Initialize TFT display
  initDisplay();
  
  // Show boot animation
  showBootAnimation();
  
  // Connect to WiFi
  connectToWiFi();
  
  // Initialize result structure
  currentResult.status = "waiting";
  currentResult.topMaterial = "Initializing";
  currentResult.confidence = 0.0;
  currentResult.extinguisher = "Unknown";
  currentResult.riskLevel = "None";
  currentResult.materialCount = 0;
  currentResult.hasNewData = false;
  
  // Show startup screen
  showStartupScreen();
  
  Serial.println("ESP32 Flame Analyzer Display Ready!");
}

void loop() {
  // Check WiFi connection
  if (WiFi.status() != WL_CONNECTED) {
    wifiConnected = false;
    showWiFiError();
    connectToWiFi();
    return;
  } else {
    wifiConnected = true;
  }
  
  // Update data from API
  if (millis() - lastUpdate >= UPDATE_INTERVAL) {
    fetchFlameData();
    lastUpdate = millis();
  }
  
  // Switch display modes automatically
  if (millis() - modeSwitch >= MODE_SWITCH_INTERVAL) {
    displayMode = (displayMode + 1) % 3;
    modeSwitch = millis();
    currentResult.hasNewData = true; // Force redraw
  }
  
  // Update display if new data available
  if (currentResult.hasNewData) {
    updateDisplay();
    currentResult.hasNewData = false;
  }
  
  delay(100);
}

void initDisplay() {
  Serial.println("Initializing TFT display...");
  
  // Initialize display with hardware SPI
  tft.initR(INITR_BLACKTAB);  // Use for 1.8" ST7735R displays
  tft.setRotation(3);  // Landscape orientation
  tft.fillScreen(BLACK);
  
  // Display dimensions are now 160x128 in landscape
  Serial.println("TFT display initialized (160x128)");
}

void connectToWiFi() {
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
    
    // Show connection progress on display
    tft.fillScreen(BLACK);
    tft.setTextSize(1);
    tft.setTextColor(CYAN);
    tft.setCursor(10, 30);
    tft.println("Connecting WiFi...");
    tft.setCursor(10, 50);
    tft.println(ssid);
    tft.setCursor(10, 70);
    for (int i = 0; i < attempts; i++) {
      tft.print(".");
    }
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("");
    Serial.println("WiFi connected!");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
    
    // Auto-detect server IP (same subnet)
    String localIP = WiFi.localIP().toString();
    int lastDot = localIP.lastIndexOf('.');
    serverIP = localIP.substring(0, lastDot + 1) + "100";  // Assume PC is .100
    Serial.println("Server IP set to: " + serverIP);
    
    wifiConnected = true;
  } else {
    Serial.println("WiFi connection failed!");
    wifiConnected = false;
  }
}

void showBootAnimation() {
  tft.fillScreen(BLACK);
  
  // Animated title appearance
  for (int i = 0; i < 5; i++) {
    tft.setTextColor(random(0x1000, 0xFFFF));
    tft.setTextSize(1);
    tft.setCursor(20 + random(-5, 5), 40 + random(-2, 2));
    tft.println("FLAME ANALYZER");
    delay(200);
    tft.fillRect(15, 35, 130, 20, BLACK);
  }
  
  // Final stable title
  tft.setTextColor(ORANGE);
  tft.setCursor(20, 40);
  tft.println("FLAME ANALYZER");
  
  delay(1000);
}

void showStartupScreen() {
  tft.fillScreen(BLACK);
  
  // Title
  tft.setTextSize(1);
  tft.setTextColor(ORANGE);
  tft.setCursor(20, 10);
  tft.println("FLAME ANALYZER");
  
  // Version
  tft.setTextColor(WHITE);
  tft.setCursor(55, 25);
  tft.println("v1.0 ESP32");
  
  // Fire emoji effect
  tft.setTextColor(RED);
  tft.setTextSize(2);
  tft.setCursor(70, 45);
  tft.println("FIRE");
  
  // Status
  tft.setTextSize(1);
  tft.setTextColor(GREEN);
  tft.setCursor(10, 80);
  tft.println("WiFi: " + String(wifiConnected ? "Connected" : "Connecting"));
  tft.setCursor(10, 95);
  tft.println("API: " + serverIP + ":5000");
  
  delay(2000);
}

void showWiFiError() {
  tft.fillScreen(RED);
  tft.setTextSize(1);
  tft.setTextColor(WHITE);
  tft.setCursor(20, 40);
  tft.println("WiFi ERROR!");
  tft.setCursor(10, 60);
  tft.println("Reconnecting...");
}

void fetchFlameData() {
  if (!wifiConnected) return;
  
  HTTPClient http;
  String url = "http://" + serverIP + ":" + String(serverPort) + apiEndpoint;
  
  Serial.println("Fetching data from: " + url);
  http.begin(url);
  http.setTimeout(5000);
  
  int httpCode = http.GET();
  
  if (httpCode == HTTP_CODE_OK) {
    String payload = http.getString();
    Serial.println("Received: " + payload);
    
    // Parse JSON response
    DynamicJsonDocument doc(2048);
    DeserializationError error = deserializeJson(doc, payload);
    
    if (!error) {
      // Update current result - FIXED: Use .as<String>() for proper conversion
      String newStatus = doc["status"].as<String>();
      if (newStatus != currentResult.status || 
          doc["summary"]["top_material"].as<String>() != currentResult.topMaterial) {
        currentResult.hasNewData = true;
      }
      
      currentResult.status = newStatus;
      currentResult.topMaterial = doc["summary"]["top_material"].as<String>();
      currentResult.confidence = doc["summary"]["confidence"];
      currentResult.extinguisher = doc["summary"]["extinguisher"].as<String>();
      currentResult.riskLevel = doc["summary"]["risk_level"].as<String>();
      currentResult.timestamp = doc["timestamp"];
      
      // Parse materials array
      JsonArray materials = doc["materials"];
      currentResult.materialCount = min((int)materials.size(), 3);
      
      for (int i = 0; i < currentResult.materialCount; i++) {
        currentResult.materials[i] = materials[i]["name"].as<String>();
        currentResult.similarities[i] = materials[i]["similarity"];
      }
      
      Serial.println("Data updated: " + currentResult.topMaterial + 
                    " (" + String(currentResult.confidence, 1) + "%)");
    } else {
      Serial.println("JSON parsing error: " + String(error.c_str()));
    }
  } else {
    Serial.println("HTTP error: " + String(httpCode));
  }
  
  http.end();
}

uint16_t getStatusColor(String status) {
  if (status == "complete") return GREEN;
  if (status == "analyzing") return YELLOW;
  if (status == "no_fire") return BLUE;
  if (status == "error") return RED;
  return GRAY;
}

String getCurrentTime() {
  unsigned long uptime = millis() / 1000;
  int minutes = (uptime / 60) % 60;
  int seconds = uptime % 60;
  return String(minutes) + ":" + (seconds < 10 ? "0" : "") + String(seconds);
}

uint16_t getRiskColor(String risk) {
  if (risk == "High" || risk.toFloat() > 80) return RED;
  if (risk == "Medium" || risk.toFloat() > 50) return ORANGE;
  if (risk == "Low" || risk.toFloat() > 20) return YELLOW;
  return GREEN;
}

void updateDisplay() {
  switch (displayMode) {
    case 0:
      showMainDisplay();
      break;
    case 1:
      showDetailDisplay();
      break;
    case 2:
      showMaterialsList();
      break;
  }
}

void showMainDisplay() {
  tft.fillScreen(BLACK);
  
  // Header
  tft.setTextSize(1);
  tft.setTextColor(ORANGE);
  tft.setCursor(5, 5);
  tft.println("FLAME ANALYZER");
  
  // Status indicator
  uint16_t statusColor = getStatusColor(currentResult.status);
  tft.fillCircle(145, 10, 6, statusColor);
  
  // Main content based on status
  if (currentResult.status == "no_fire") {
    showNoFireScreen();
  } else if (currentResult.status == "complete" && currentResult.materialCount > 0) {
    showFireDetectedScreen();
  } else if (currentResult.status == "no_data" || currentResult.status == "waiting") {
    showWaitingScreen();
  } else {
    showAnalyzingScreen();
  }
  
  // Footer with mode indicator
  tft.setTextSize(1);
  tft.setTextColor(GRAY);
  tft.setCursor(5, 115);
  tft.println("Mode:" + String(displayMode + 1) + "/3");
  tft.setCursor(80, 115);
  tft.println(getCurrentTime());
}

void showNoFireScreen() {
  // Large OK symbol
  tft.setTextSize(3);
  tft.setTextColor(GREEN);
  tft.setCursor(65, 35);
  tft.println("OK");
  
  // Status text
  tft.setTextSize(1);
  tft.setTextColor(GREEN);
  tft.setCursor(45, 65);
  tft.println("NO FIRE");
  
  tft.setTextColor(WHITE);
  tft.setCursor(30, 80);
  tft.println("System Safe");
  
  tft.setCursor(25, 95);
  tft.println("Risk: None");
}

void showWaitingScreen() {
  tft.setTextSize(2);
  tft.setTextColor(CYAN);
  tft.setCursor(20, 35);
  tft.println("WAITING");
  
  tft.setTextSize(1);
  tft.setTextColor(WHITE);
  tft.setCursor(15, 60);
  tft.println("Monitoring for");
  tft.setCursor(25, 75);
  tft.println("flame images...");
  
  // Animated dots
  static int dotCount = 0;
  tft.setCursor(40, 90);
  for (int i = 0; i < dotCount; i++) {
    tft.print(".");
  }
  dotCount = (dotCount + 1) % 4;
}

void showAnalyzingScreen() {
  tft.setTextSize(1);
  tft.setTextColor(YELLOW);
  tft.setCursor(30, 30);
  tft.println("ANALYZING");
  
  tft.setTextColor(WHITE);
  tft.setCursor(20, 50);
  tft.println("Processing flame");
  tft.setCursor(35, 65);
  tft.println("properties...");
  
  // Progress bar effect
  static int progress = 0;
  tft.drawRect(20, 85, 120, 8, WHITE);
  tft.fillRect(22, 87, progress, 4, ORANGE);
  progress = (progress + 2) % 116;
}

void showFireDetectedScreen() {
  // Fire icon
  tft.setTextSize(2);
  tft.setTextColor(RED);
  tft.setCursor(5, 25);
  tft.println("FIRE!");
  
  // Material name (truncated if too long)
  tft.setTextSize(1);
  tft.setTextColor(YELLOW);
  tft.setCursor(70, 30);
  String material = currentResult.topMaterial;
  if (material.length() > 12) {
    material = material.substring(0, 9) + "...";
  }
  tft.println(material);
  
  // Confidence - FIXED: Complete the line
  tft.setTextColor(WHITE);
  tft.setCursor(70, 45);
  tft.println(String(currentResult.confidence, 1) + "%");
  
  // Risk level
  uint16_t riskColor = getRiskColor(currentResult.riskLevel);
  tft.setTextColor(riskColor);
  tft.setCursor(5, 60);
  tft.println("Risk: " + currentResult.riskLevel);
  
  // Extinguisher
  tft.setTextColor(CYAN);
  tft.setCursor(5, 75);
  tft.println("Use: ");
  tft.setTextColor(WHITE);
  tft.setCursor(35, 75);
  String ext = currentResult.extinguisher;
  if (ext.length() > 15) {
    ext = ext.substring(0, 12) + "...";
  }
  tft.println(ext);
  
  // Material count indicator
  tft.setTextColor(PURPLE);
  tft.setCursor(5, 95);
  tft.println("Materials: " + String(currentResult.materialCount));
}

void showDetailDisplay() {
  tft.fillScreen(BLACK);
  
  // Header
  tft.setTextSize(1);
  tft.setTextColor(ORANGE);
  tft.setCursor(5, 5);
  tft.println("FLAME DETAILS");
  
  // Status indicator
  uint16_t statusColor = getStatusColor(currentResult.status);
  tft.fillCircle(145, 10, 6, statusColor);
  
  if (currentResult.status == "complete" && currentResult.materialCount > 0) {
    // Top material with larger text
    tft.setTextSize(1);
    tft.setTextColor(YELLOW);
    tft.setCursor(5, 25);
    tft.println("TOP MATCH:");
    
    tft.setTextColor(WHITE);
    tft.setTextSize(1);
    tft.setCursor(5, 40);
    String topMat = currentResult.topMaterial;
    if (topMat.length() > 20) {
      tft.println(topMat.substring(0, 20));
      tft.setCursor(5, 55);
      tft.println(topMat.substring(20));
    } else {
      tft.println(topMat);
    }
    
    // Confidence bar
    tft.setTextColor(CYAN);
    tft.setCursor(5, 75);
    tft.println("Confidence:");
    
    // Draw confidence bar
    int barWidth = (int)(currentResult.confidence * 1.5);  // Scale to fit screen
    tft.drawRect(5, 90, 150, 10, WHITE);
    uint16_t confColor = currentResult.confidence > 70 ? GREEN : 
                        currentResult.confidence > 40 ? YELLOW : RED;
    tft.fillRect(7, 92, barWidth, 6, confColor);
    
    // Confidence percentage
    tft.setTextColor(WHITE);
    tft.setCursor(5, 105);
    tft.println(String(currentResult.confidence, 1) + "%");
  } else {
    tft.setTextColor(GRAY);
    tft.setCursor(30, 60);
    tft.println("No details available");
  }
  
  // Footer
  tft.setTextSize(1);
  tft.setTextColor(GRAY);
  tft.setCursor(5, 115);
  tft.println("Details Mode");
  tft.setCursor(80, 115);
  tft.println(getCurrentTime());
}

void showMaterialsList() {
  tft.fillScreen(BLACK);
  
  // Header
  tft.setTextSize(1);
  tft.setTextColor(ORANGE);
  tft.setCursor(5, 5);
  tft.println("MATERIALS LIST");
  
  // Status indicator
  uint16_t statusColor = getStatusColor(currentResult.status);
  tft.fillCircle(145, 10, 6, statusColor);
  
  if (currentResult.materialCount > 0) {
    // List materials
    int yPos = 25;
    for (int i = 0; i < currentResult.materialCount && i < 3; i++) {
      // Material number
      tft.setTextColor(CYAN);
      tft.setCursor(5, yPos);
      tft.println(String(i + 1) + ".");
      
      // Material name
      tft.setTextColor(WHITE);
      tft.setCursor(20, yPos);
      String matName = currentResult.materials[i];
      if (matName.length() > 18) {
        matName = matName.substring(0, 15) + "...";
      }
      tft.println(matName);
      
      // Similarity percentage
      tft.setTextColor(getRiskColor(String(currentResult.similarities[i], 0)));
      tft.setCursor(130, yPos);
      tft.println(String(currentResult.similarities[i], 0) + "%");
      
      yPos += 15;
    }
    
    // Extinguisher recommendation
    tft.setTextColor(GREEN);
    tft.setCursor(5, yPos + 10);
    tft.println("EXTINGUISHER:");
    tft.setTextColor(WHITE);
    tft.setCursor(5, yPos + 25);
    String ext = currentResult.extinguisher;
    if (ext.length() > 20) {
      tft.println(ext.substring(0, 20));
      if (yPos + 40 < 110) {  // Check if there's space for second line
        tft.setCursor(5, yPos + 40);
        tft.println(ext.substring(20));
      }
    } else {
      tft.println(ext);
    }
  } else {
    tft.setTextColor(GRAY);
    tft.setCursor(25, 60);
    tft.println("No materials detected");
  }
  
  // Footer
  tft.setTextSize(1);
  tft.setTextColor(GRAY);
  tft.setCursor(5, 115);
  tft.println("Materials Mode");
  tft.setCursor(80, 115);
  tft.println(getCurrentTime());
}

// Animation functions for dynamic display
void showFlameAnimation() {
  static int frame = 0;
  static unsigned long lastFrame = 0;
  
  if (millis() - lastFrame > 500) {  // 2 FPS animation
    // Simple flame flicker effect
    uint16_t flameColors[] = {RED, ORANGE, YELLOW, RED};
    uint16_t currentFlameColor = flameColors[frame % 4];
    
    // Draw animated flame icon
    tft.fillTriangle(75, 35, 85, 35, 80, 25, currentFlameColor);
    tft.fillTriangle(70, 45, 90, 45, 80, 30, currentFlameColor);
    
    frame++;
    lastFrame = millis();
  }
}

void showConnectionStatus() {
  // WiFi status
  tft.setTextColor(wifiConnected ? GREEN : RED);
  tft.setCursor(5, 20);
  tft.println("WiFi: " + String(wifiConnected ? "OK" : "ERR"));
  
  // Server status (if last update was recent)
  bool serverOk = (millis() - lastUpdate) < UPDATE_INTERVAL * 2;
  tft.setTextColor(serverOk ? GREEN : RED);
  tft.setCursor(80, 20);
  tft.println("API: " + String(serverOk ? "OK" : "ERR"));
}

void displaySystemInfo() {
  tft.fillScreen(BLACK);
  
  // Title
  tft.setTextSize(1);
  tft.setTextColor(CYAN);
  tft.setCursor(30, 5);
  tft.println("SYSTEM INFO");
  
  // WiFi info
  tft.setTextColor(WHITE);
  tft.setCursor(5, 25);
  tft.println("SSID: " + String(ssid));
  tft.setCursor(5, 40);
  tft.println("IP: " + WiFi.localIP().toString());
  
  // Server info
  tft.setCursor(5, 55);
  tft.println("Server: " + serverIP);
  tft.setCursor(5, 70);
  tft.println("Port: " + String(serverPort));
  
  // Memory info
  tft.setCursor(5, 85);
  tft.println("Free RAM: " + String(ESP.getFreeHeap()));
  
  // Uptime
  tft.setCursor(5, 100);
  tft.println("Uptime: " + String(millis() / 1000) + "s");
}

void showProgressBar(int progress, String text) {
  tft.setTextColor(WHITE);
  tft.setCursor(5, 50);
  tft.println(text);
  
  // Progress bar
  tft.drawRect(5, 70, 150, 12, WHITE);
  int fillWidth = (progress * 146) / 100;
  tft.fillRect(7, 72, fillWidth, 8, GREEN);
  
  // Percentage
  tft.setCursor(70, 90);
  tft.println(String(progress) + "%");
}

void showErrorScreen(String errorMsg) {
  tft.fillScreen(RED);
  tft.setTextSize(1);
  tft.setTextColor(WHITE);
  tft.setCursor(10, 40);
  tft.println("ERROR:");
  tft.setCursor(10, 60);
  
  // Word wrap error message
  int maxCharsPerLine = 20;
  int pos = 0;
  int line = 0;
  while (pos < errorMsg.length() && line < 3) {
    int endPos = min(pos + maxCharsPerLine, (int)errorMsg.length());
    String lineText = errorMsg.substring(pos, endPos);
    tft.setCursor(10, 60 + (line * 15));
    tft.println(lineText);
    pos = endPos;
    line++;
  }
}

// Utility functions for enhanced display features
void drawBattery(int x, int y, int percentage) {
  // Battery outline
  tft.drawRect(x, y, 20, 10, WHITE);
  tft.drawRect(x + 20, y + 3, 2, 4, WHITE);
  
  // Battery fill
  int fillWidth = (percentage * 18) / 100;
  uint16_t battColor = percentage > 50 ? GREEN : percentage > 20 ? YELLOW : RED;
  tft.fillRect(x + 1, y + 1, fillWidth, 8, battColor);
}

void drawSignalStrength(int x, int y, int strength) {
  // Signal bars (1-4 bars based on strength)
  int bars = (strength + 25) / 25;  // Convert 0-100 to 1-4 bars
  
  for (int i = 0; i < 4; i++) {
    uint16_t barColor = i < bars ? GREEN : GRAY;
    int barHeight = 3 + (i * 2);
    tft.fillRect(x + (i * 4), y + (8 - barHeight), 3, barHeight, barColor);
  }
}
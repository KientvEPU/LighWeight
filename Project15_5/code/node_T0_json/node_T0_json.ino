#include <SPI.h>
#include <LoRa.h>

// Định nghĩa các chân kết nối LoRa
#define LORA_SCK 13
#define LORA_MISO 12
#define LORA_MOSI 11
#define LORA_SS 10
#define LORA_RST 9
#define LORA_DIO0 2

#define LORA_FREQUENCY 433E6

void setup() {
  Serial.begin(9600);
  while (!Serial);
  Serial.println("LoRa Central Node T0 - Receiver");

  LoRa.setPins(LORA_SS, LORA_RST, LORA_DIO0);

  if (!LoRa.begin(LORA_FREQUENCY)) {
    Serial.println("Starting LoRa failed!");
    while (1);
  }

  LoRa.setSyncWord(0xF3);
  Serial.println("LoRa initialized successfully. Waiting for packets...");
}

void loop() {
  int packetSize = LoRa.parsePacket();
  if (packetSize) {
    String receivedData = "";
    while (LoRa.available()) {
      receivedData += (char)LoRa.read();
    }

    Serial.print("Raw data received: ");
    Serial.println(receivedData);

    // Tách chuỗi nhận được theo dấu phẩy
    int firstComma = receivedData.indexOf(',');
    int secondComma = receivedData.indexOf(',', firstComma + 1);
    int thirdComma = receivedData.indexOf(',', secondComma + 1);

    if (firstComma > 0 && secondComma > firstComma && thirdComma > secondComma) {
      String nodeID_str = receivedData.substring(0, firstComma); // A0, A1,...
      String temp_str = receivedData.substring(firstComma + 1, secondComma);
      String hum_str = receivedData.substring(secondComma + 1, thirdComma);
      String pir_str = receivedData.substring(thirdComma + 1);

      // Chuyển node ID sang số (A0 -> 0, A1 -> 1,...)
      int nodeNum = -1;
      if (nodeID_str.charAt(0) == 'A') {
        nodeNum = nodeID_str.substring(1).toInt();
      }

      if (nodeNum >= 0) {
        // Tạo chuỗi JSON
        String json = "{\"id\":" + String(nodeNum);
        json += ",\"t\":" + temp_str;
        json += ",\"h\":" + hum_str;
        json += ",\"p\":" + pir_str + "}";

        // In ra Serial 232 (UART)
        Serial.println(json);
      } else {
        Serial.println("Invalid node ID format");
      }
    } else {
      Serial.println("Invalid data format");
    }
  }
}

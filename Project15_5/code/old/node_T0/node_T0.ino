#include <SPI.h>
#include <LoRa.h>

// Định nghĩa các chân kết nối LoRa
#define LORA_SCK 13   // SCK của LoRa đến D13 của Arduino
#define LORA_MISO 12  // MISO của LoRa đến D12 của Arduino
#define LORA_MOSI 11  // MOSI của LoRa đến D11 của Arduino
#define LORA_SS 10    // NSS (SS) của LoRa đến D10 của Arduino (Slave Select)
#define LORA_RST 9    // RST của LoRa đến D9 của Arduino (Reset)
#define LORA_DIO0 2   // DIO0 của LoRa đến D2 của Arduino (Chân ngắt)

// Tần số LoRa (PHẢI GIỐNG NHAU TRÊN TẤT CẢ CÁC NODE)
#define LORA_FREQUENCY 433E6 // Ví dụ: 433MHz. Sử dụng 868E6 cho 868MHz, 915E6 cho 915MHz

void setup() {
  Serial.begin(9600); // Khởi tạo giao tiếp Serial với PC
  while (!Serial);    // Đợi cổng Serial kết nối (cần thiết cho một số board Arduino)
  Serial.println("LoRa Central Node T0 - Receiver");

  // Khởi tạo module LoRa
  LoRa.setPins(LORA_SS, LORA_RST, LORA_DIO0); // Thiết lập các chân cho thư viện LoRa
  // Một số board có thể cần khai báo tường minh các chân SPI:
  // SPI.begin(LORA_SCK, LORA_MISO, LORA_MOSI, LORA_SS);


  if (!LoRa.begin(LORA_FREQUENCY)) {
    Serial.println("Starting LoRa failed!");
    while (1); // Dừng nếu không khởi tạo được LoRa
  }
  Serial.print("LoRa initialized successfully at ");
  Serial.print(LORA_FREQUENCY / 1E6);
  Serial.println(" MHz");

  LoRa.setSyncWord(0xF3); // Đặt Sync Word để tránh nhiễu từ các mạng LoRa khác (tùy chọn nhưng khuyến nghị)
                          // Sync Word phải giống nhau trên tất cả các node.
  Serial.println("Waiting for LoRa packets...");
}

void loop() {
  // Cố gắng phân tích một gói tin LoRa
  int packetSize = LoRa.parsePacket();
  if (packetSize) {
    // Đã nhận được một gói tin
    Serial.print("Received packet '");

    String receivedData = "";
    while (LoRa.available()) {
      receivedData += (char)LoRa.read();
    }

    Serial.print(receivedData);
    Serial.print("' with RSSI "); // RSSI: Received Signal Strength Indicator
    Serial.println(LoRa.packetRssi());

    // Gửi dữ liệu nhận được đến PC qua cổng Serial
    // Dữ liệu đã được định dạng sẵn từ các node gửi: "NODE_ID,TEMP,HUM,PIR_STATUS"
    Serial.println(receivedData); 
  }
}
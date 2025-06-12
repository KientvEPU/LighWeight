#include <SPI.h>
#include <LoRa.h>
#include <DHT.h>

// Cấu hình Node
#define NODE_ID "A0"

// Định nghĩa các chân kết nối LoRa
#define LORA_SCK 13
#define LORA_MISO 12
#define LORA_MOSI 11
#define LORA_SS 10
#define LORA_RST 9
#define LORA_DIO0 2


// Tần số LoRa (PHẢI GIỐNG NODE T0)
#define LORA_FREQUENCY 433E6

// Cấu hình cảm biến DHT
#define DHTPIN 3       // Chân Digital kết nối với cảm biến DHT (ví dụ D3)
#define DHTTYPE DHT11  // Loại cảm biến: DHT22 (AM2302). Sử dụng DHT11 nếu bạn dùng loại đó.
DHT dht(DHTPIN, DHTTYPE);

// Cấu hình cảm biến PIR
#define PIRPIN 4       // Chân Digital kết nối với output của cảm biến PIR (ví dụ D4)

// Khoảng thời gian gửi dữ liệu (miligiây)
#define SEND_INTERVAL 2000 // Gửi dữ liệu mỗi 2 giây
// #define SEND_ON_MOTION_ONLY // Bỏ comment dòng này nếu chỉ muốn gửi khi có chuyển động (và định kỳ)

unsigned long lastSendTime = 0;
int lastPirState = LOW; // Để theo dõi sự thay đổi trạng thái PIR

void setup() {
  Serial.begin(9600);
  while (!Serial);
  Serial.print("LoRa Sensor Node: ");
  Serial.println(NODE_ID);

  // Khởi tạo module LoRa
  LoRa.setPins(LORA_SS, LORA_RST, LORA_DIO0);
  // SPI.begin(LORA_SCK, LORA_MISO, LORA_MOSI, LORA_SS); // Tùy chọn cho một số board

  if (!LoRa.begin(LORA_FREQUENCY)) {
    Serial.println("Starting LoRa failed!");
    while (1);
  }
  Serial.print("LoRa initialized successfully at ");
  Serial.print(LORA_FREQUENCY / 1E6);
  Serial.println(" MHz");
  LoRa.setSyncWord(0xF3); // Phải giống Sync Word của Node T0

  // Khởi tạo cảm biến DHT
  dht.begin();
  Serial.println("DHT sensor initialized.");

  // Khởi tạo cảm biến PIR
  pinMode(PIRPIN, INPUT);
  Serial.println("PIR sensor initialized.");
  
  // Đợi cảm biến ổn định một chút, đặc biệt là PIR
  Serial.println("Calibrating PIR sensor...");
  delay(10000); // Chờ 10 giây cho PIR ổn định. Một số PIR cần thời gian dài hơn.
  Serial.println("PIR sensor calibrated.");

  delay(1000); // Đợi các cảm biến khác ổn định
}

void loop() {
  unsigned long currentTime = millis();
  bool shouldSend = false;

  // Đọc trạng thái cảm biến PIR
  int currentPirState = digitalRead(PIRPIN);

#ifdef SEND_ON_MOTION_ONLY
  if (currentPirState == HIGH && lastPirState == LOW) { // Chỉ gửi khi phát hiện chuyển động mới
    Serial.println("Motion detected!");
    shouldSend = true;
  }
  lastPirState = currentPirState; // Cập nhật trạng thái PIR trước đó

  // Gửi định kỳ ngay cả khi không có chuyển động để đảm bảo node vẫn hoạt động
  if (!shouldSend && (currentTime - lastSendTime > SEND_INTERVAL * 2)) { // Ví dụ gửi mỗi 60 giây nếu không có chuyển động
     Serial.println("Periodic send (no motion change).");
     shouldSend = true;
  }
#else
  if (currentTime - lastSendTime > SEND_INTERVAL) {
    shouldSend = true;
  }
#endif

  if (shouldSend) {
    // Đọc dữ liệu từ cảm biến DHT
    float humidity = dht.readHumidity();
    // Đọc nhiệt độ bằng Celsius (mặc định)
    float temperature = dht.readTemperature();
    // Đọc nhiệt độ bằng Fahrenheit (nếu cần): float temperature = dht.readTemperature(true);

    // Kiểm tra nếu đọc lỗi (ví dụ: trả về NaN - Not a Number)
    if (isnan(humidity) || isnan(temperature)) {
      Serial.println("Failed to read from DHT sensor!");
      lastSendTime = currentTime; // Reset timer để tránh gửi liên tục khi lỗi
      return; // Thoát khỏi lần lặp này và thử lại sau
    }

    // Tạo chuỗi gói tin dữ liệu
    // Định dạng: NODE_ID,TEMPERATURE,HUMIDITY,PIR_STATUS
    String dataPacket = String(NODE_ID) + ",";
    dataPacket += String(temperature, 1) + ","; // Nhiệt độ, 1 chữ số thập phân
    dataPacket += String(humidity, 1) + ",";    // Độ ẩm, 1 chữ số thập phân
    dataPacket += String(currentPirState);      // Trạng thái PIR (0 hoặc 1)

    // Gửi gói tin LoRa
    Serial.print("Sending packet: ");
    Serial.println(dataPacket);

    LoRa.beginPacket(); // Bắt đầu một gói tin mới
    LoRa.print(dataPacket);
    int transmissionState = LoRa.endPacket(); // Kết thúc và gửi gói tin, trả về trạng thái gửi

    if (transmissionState) { // transmissionState == 1 (true) nghĩa là gửi thành công
        Serial.println("Packet sent successfully!");
    } else { // transmissionState == 0 (false) nghĩa là gửi thất bại
        Serial.println("Failed to send packet. LoRa busy or other error.");
    }
    
    lastSendTime = currentTime; // Cập nhật thời gian gửi cuối cùng
  }
}
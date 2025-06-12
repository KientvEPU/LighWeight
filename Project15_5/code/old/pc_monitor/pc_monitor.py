import serial
import time

# Cấu hình cổng Serial và Baud rate
# THAY THẾ 'COM5' BẰNG CỔNG COM CHÍNH XÁC CỦA ARDUINO T0 TRÊN MÁY BẠN
# Trên Windows: 'COM3', 'COM4', ...
# Trên Linux: '/dev/ttyUSB0', '/dev/ttyACM0', ...
# Trên macOS: '/dev/cu.usbserial-xxxx' hoặc tương tự
SERIAL_PORT = 'COM5'  # <<<----- THAY ĐỔI CỔNG NÀY CHO PHÙ HỢP
BAUD_RATE = 9600

def parse_data(data_line):
    """Phân tích chuỗi dữ liệu từ Arduino."""
    parts = data_line.split(',')
    if len(parts) == 4:
        node_id = parts[0]
        try:
            temperature = float(parts[1])
            humidity = float(parts[2])
            pir_status_val = int(parts[3])
            pir_status_text = "Có chuyển động" if pir_status_val == 1 else "Không có chuyển động"
            
            print(f"Node ID: {node_id}")
            print(f"  Nhiệt độ: {temperature:.1f}°C")
            print(f"  Độ ẩm: {humidity:.1f}%")
            print(f"  PIR: {pir_status_text} ({pir_status_val})")
            print("-" * 30)
            
            # Ở đây bạn có thể thêm code để lưu vào file CSV hoặc database
            # save_to_csv(node_id, temperature, humidity, pir_status_val)

        except ValueError:
            print(f"Lỗi phân tích dữ liệu: '{data_line}'. Định dạng không đúng.")
        except IndexError:
            print(f"Lỗi phân tích dữ liệu: '{data_line}'. Thiếu trường dữ liệu.")
    # Bỏ qua các dòng log hệ thống từ T0
    elif "Received packet" in data_line or \
         "LoRa Central Node T0" in data_line or \
         "Waiting for LoRa packets" in data_line or \
         "LoRa initialized successfully" in data_line:
        print(f"Thông báo từ T0: {data_line}") # In ra để debug nếu cần
    elif data_line: # Nếu dòng không rỗng và không khớp các mẫu trên
        print(f"Dữ liệu không xác định từ T0: {data_line}")


def save_to_csv(node_id, temp, hum, pir):
    """Ví dụ hàm lưu dữ liệu vào file CSV."""
    filename = "sensor_data.csv"
    try:
        with open(filename, "a") as f: # "a" để ghi nối tiếp vào file
            # Ghi header nếu file mới được tạo hoặc trống
            import os
            if os.path.getsize(filename) == 0:
                f.write("Timestamp,NodeID,Temperature,Humidity,PIR_Status\n")
            
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp},{node_id},{temp},{hum},{pir}\n")
    except Exception as e:
        print(f"Lỗi khi ghi file CSV: {e}")


def main():
    print(f"Đang cố gắng kết nối tới cổng {SERIAL_PORT} với baud rate {BAUD_RATE}...")
    try:
        # timeout=1 để readline không bị block vô hạn nếu không có dữ liệu
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Đã kết nối tới {SERIAL_PORT}. Đang chờ dữ liệu từ Node T0...")
    except serial.SerialException as e:
        print(f"Lỗi: Không thể mở cổng serial {SERIAL_PORT}. {e}")
        print("Vui lòng kiểm tra lại tên cổng và đảm bảo Node T0 (Arduino) đã được kết nối.")
        return

    try:
        while True:
            if ser.in_waiting > 0: # Kiểm tra xem có dữ liệu trong buffer không
                try:
                    # Đọc một dòng dữ liệu từ cổng Serial, decode từ byte sang utf-8 và loại bỏ ký tự xuống dòng
                    line = ser.readline().decode('utf-8').strip()
                    if line: # Nếu dòng không rỗng
                        parse_data(line)
                except UnicodeDecodeError:
                    # Đôi khi có thể nhận được byte không hợp lệ, bỏ qua chúng
                    print("Lỗi decode Unicode. Bỏ qua dòng dữ liệu lỗi.")
                except Exception as e:
                    print(f"Lỗi khi đọc hoặc xử lý dữ liệu: {e}")
            
            # time.sleep(0.1) # Có thể thêm một khoảng dừng nhỏ để giảm tải CPU

    except KeyboardInterrupt:
        print("\nĐang thoát chương trình.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print(f"Đã đóng cổng serial {SERIAL_PORT}.")

if __name__ == "__main__":
    main()

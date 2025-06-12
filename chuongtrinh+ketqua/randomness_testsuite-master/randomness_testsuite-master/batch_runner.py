import os
import argparse
import tkinter as tk
from Main import Main  # Đảm bảo main.py nằm cùng thư mục hoặc trong PYTHONPATH


def process_directory(input_dir, output_dir):
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    if not os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' not found, creating it.")
        os.makedirs(output_dir)

    # Tạo một cửa sổ gốc Tk ẩn. Điều này cần thiết vì lớp Main và các thành phần của nó
    # là các widget Tkinter và mong đợi một master.
    try:
        root = tk.Tk()
        root.withdraw()  # Ẩn cửa sổ gốc
    except tk.TclError as e:
        print(f"Error initializing Tkinter: {e}")
        print(
            "This script might need a display environment (like X11) to run, even in batch mode, due to Tkinter dependencies.")
        print("Try running with 'xvfb-run python batch_runner.py ...' if you are on a headless server.")
        return

    # Khởi tạo lớp Main với batch_mode=True
    # Điều này giả định main.py đã được sửa đổi để chấp nhận batch_mode
    app = Main(master=root, batch_mode=True)

    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]

    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        root.destroy()
        return

    print(f"Found {len(txt_files)} .txt files to process in '{input_dir}'.")

    for filename in txt_files:
        input_file_path = os.path.join(input_dir, filename)
        # Xây dựng tên tệp đầu ra duy nhất
        base, ext = os.path.splitext(filename)
        output_filename = f"{base}_results{ext}"  # Ví dụ: GF1_gm1_final_PN_results.txt
        output_file_path = os.path.join(output_dir, output_filename)

        print(f"\nProcessing file: {input_file_path}")

        # 1. Reset trạng thái ứng dụng cho tệp mới (quan trọng!)
        app.reset()  # Xóa các đầu vào, kết quả và lựa chọn kiểm tra trước đó

        # 2. Đặt đường dẫn tệp đầu vào theo chương trình
        # Điều này mô phỏng select_binary_file() mà không mở hộp thoại
        app._Main__file_name = input_file_path  # Đặt thủ công biến riêng __file_name

        # Giả sử các đối tượng input field này được tạo trong init_window()
        if hasattr(app, '_Main__binary_data_file_input') and app._Main__binary_data_file_input:
            app._Main__binary_data_file_input.set_data(input_file_path)
        else:
            print(
                "Warning: _Main__binary_data_file_input not found or not initialized. File path might not be set correctly in GUI components.")

        if hasattr(app, '_Main__binary_input') and app._Main__binary_input:
            app._Main__binary_input.set_data('')  # Xóa đầu vào nhị phân trực tiếp

        if hasattr(app, '_Main__string_data_file_input') and app._Main__string_data_file_input:
            app._Main__string_data_file_input.set_data('')  # Xóa đầu vào tệp dữ liệu chuỗi

        app._Main__is_binary_file = True
        app._Main__is_data_file = False

        # 3. Chọn tất cả các bài kiểm tra (hoặc một bộ cụ thể nếu cần)
        app.select_all()

        # 4. Thực thi các bài kiểm tra
        # Phương thức execute bây giờ sẽ sử dụng đường dẫn tệp được đặt trong __binary_data_file_input
        # và sử dụng print cho thông báo nếu batch_mode là True
        print("Executing tests...")
        app.execute()  # Điều này sẽ điền vào app._test_result

        # 5. Lưu kết quả
        # Phương thức save_result_to_file cần được gọi với output_path
        if app._test_result:  # Kiểm tra xem execute có tạo ra kết quả nào không
            print(f"Saving results to: {output_file_path}")
            # Gọi save_result_to_file đã sửa đổi với output_path
            app.save_result_to_file(output_path=output_file_path)
        else:
            print(f"No results generated for {input_file_path}. Skipping save.")

        print(f"Finished processing {filename}")

    print(f"\nBatch processing complete. Results saved in '{output_dir}'.")

    # Hủy cửa sổ gốc một cách rõ ràng khi không cần nữa
    root.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch process randomness tests for .txt files using randomness_testsuite.")
    parser.add_argument("input_dir", help="Directory containing the .txt files with binary sequences.")
    parser.add_argument("output_dir", help="Directory where results will be saved.")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir)
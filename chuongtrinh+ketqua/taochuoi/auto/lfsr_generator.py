import numpy as np
import os
import argparse

# Định nghĩa các đa thức được cung cấp
# Đa thức được biểu diễn dưới dạng np.array([c_m, c_{m-1}, ..., c_1, c_0])
# trong đó c_k là hệ số của x^k, và c_m luôn là 1.

POLYNOMIALS_DEG12 = [
    np.array([1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1]),  # x^12+x^6+x^4+x^1+1
    np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1]),  # x^12+x^9+x^3+x^2+1
    np.array([1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1]),  # x^12+x^9+x^8+x^3+x^2+1
    np.array([1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1]),  # x^12+x^10+x^9+x^8+x^6+x^2+1
    np.array([1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1]),  # x^12+x^10+x^9+x^8+x^6+x^5+x^4+x^2+1
    np.array([1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1]),  # x^12+x^11+x^6+x^4+x^2+x^1+1
    np.array([1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1]),  # x^12+x^11+x^9+x^5+x^3+x^1+1
    np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1]),  # x^12+x^11+x^9+x^7+x^6+x^4+1
    np.array([1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1]),  # x^12+x^11+x^9+x^7+x^6+x^5+1
    np.array([1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1]),  # x^12+x^11+x^9+x^8+x^7+x^4+1
    np.array([1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1]),  # x^12+x^11+x^9+x^8+x^7+x^5+x^2+x^1+1
    np.array([1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1]),  # x^12+x^11+x^10+x^5+x^2+x^1+1
    np.array([1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1]),  # x^12+x^11+x^10+x^8+x^6+x^4+x^3+x^1+1
    np.array([1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1])  # x^12+x^11+x^10+x^9+x^8+x^7+x^5+x^4+x^3+x^1+1
]

POLYNOMIALS_DEG6 = [
    np.array([1, 0, 0, 0, 0, 1, 1]),  # x^6+x+1
    np.array([1, 0, 1, 1, 0, 1, 1]),  # x^6+x^4+x^3+x+1
    np.array([1, 1, 0, 0, 0, 0, 1]),  # x^6+x^5+1
    np.array([1, 1, 0, 0, 1, 1, 1]),  # x^6+x^5+x^2+x+1
    np.array([1, 1, 0, 1, 1, 0, 1]),  # x^6+x^5+x^3+x^2+1
    np.array([1, 1, 1, 0, 0, 1, 1])  # x^6+x^5+x^4+x+1
]

ALL_POLYNOMIALS = POLYNOMIALS_DEG12 + POLYNOMIALS_DEG6


def get_tap_powers_from_poly(poly_array):
    """
    Trích xuất các bậc của tap từ mảng đa thức.
    poly_array[0] là hệ số của x^m (bậc cao nhất).
    poly_array[m-j] là hệ số của x^j.
    Taps là các bậc j < m có hệ số là 1.
    """
    degree = len(poly_array) - 1
    tap_powers = []
    for power in range(degree):  # power j từ 0 đến m-1
        # Hệ số của x^power là poly_array[degree - power]
        if poly_array[degree - power] == 1:
            tap_powers.append(power)
    return sorted(tap_powers)


def lfsr_generate(degree, tap_powers, seed_initial, num_bits):
    """
    Tạo chuỗi nhị phân bằng LFSR (Fibonacci).
    State: [s_{m-1}, s_{m-2}, ..., s_1, s_0] (m bit)
    Output: s_0 (bit cuối cùng state[degree-1])
    Feedback: XOR của các bit state[degree-1-power] cho mỗi power trong tap_powers
    Bit feedback mới sẽ được dịch vào vị trí s_{m-1} (state[0])
    """
    if len(seed_initial) != degree:
        raise ValueError("Độ dài seed phải bằng bậc của LFSR.")
    if not any(s == 1 for s in seed_initial):  # all(s == 0 for s in seed_initial)
        raise ValueError("Seed không được chứa toàn số 0.")

    state = list(seed_initial)  # Tạo bản sao có thể thay đổi
    output_sequence = []

    for _ in range(num_bits):
        # Bit output là bit cuối cùng của thanh ghi (tương ứng với s_0)
        output_bit = state[degree - 1]
        output_sequence.append(output_bit)

        # Tính toán bit feedback
        feedback_val = 0
        for power in tap_powers:
            # state[k] là bit tại vị trí k.
            # state[0] là s_{m-1}, state[degree-1] là s_0.
            # Tap tại bậc 'power' (tức là s_power) nằm ở state[degree - 1 - power].
            feedback_val ^= state[degree - 1 - power]

        # Dịch phải thanh ghi: bit feedback mới vào đầu, các bit khác dịch sang
        state = [feedback_val] + state[:-1]

    return "".join(map(str, output_sequence))


def main():
    parser = argparse.ArgumentParser(description="Tạo chuỗi LFSR và lưu vào tệp .txt.")
    parser.add_argument("output_dir", help="Thư mục để lưu các tệp .txt được tạo.")
    parser.add_argument("--num_bits", type=int, default=1000000,
                        help="Số lượng bit cho mỗi chuỗi (mặc định: 1,000,000).")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Đã tạo thư mục đầu ra: {args.output_dir}")

    print(f"Sẽ tạo {len(ALL_POLYNOMIALS)} chuỗi, mỗi chuỗi dài {args.num_bits} bit.")

    for poly_idx, poly_array in enumerate(ALL_POLYNOMIALS):
        degree = len(poly_array) - 1

        poly_str_for_log = "".join(map(str, poly_array.tolist()))  # Để dễ đọc log
        print(f"\nĐang xử lý đa thức {poly_idx + 1}/{len(ALL_POLYNOMIALS)}: {poly_str_for_log} (bậc {degree})")

        try:
            tap_powers = get_tap_powers_from_poly(poly_array)
            if not tap_powers and degree > 0:  # Đa thức là x^m (không có tap nào khác)
                print(f"⚠#  Cảnh báo: Đa thức {poly_str_for_log} không có tap nào từ x^0 đến x^(m-1). "
                      "Điều này sẽ tạo ra chuỗi không hữu ích. Bỏ qua.")
                continue

            print(f"   -> Bậc: {degree}, Taps (bậc của x): {tap_powers}")

            # Seed mặc định: một bit 1 ở vị trí s_0, còn lại là 0
            # State: [s_{m-1}, ..., s_1, s_0]
            # Seed [0,0,...,1] tương ứng s_0=1, các s_i khác bằng 0.
            if degree > 0:
                seed = [0] * (degree - 1) + [1]
            else:  # Trường hợp bậc 0 (không thực tế cho LFSR nhưng để tránh lỗi)
                seed = [1]

            sequence = lfsr_generate(degree, tap_powers, seed, args.num_bits)

            # Tạo tên tệp
            taps_str = "_".join(map(str, tap_powers))
            filename = f"lfsr_deg{degree}_taps_{taps_str}_polyidx{poly_idx}.txt"
            filepath = os.path.join(args.output_dir, filename)

            with open(filepath, 'w') as f:
                f.write(sequence)
            print(f"   -> Đã lưu: {filepath}")

        except Exception as e:
            print(f"# Lỗi khi xử lý đa thức {poly_str_for_log}: {e}")

    print("\nHoàn tất quá trình tạo chuỗi LFSR.")


if __name__ == "__main__":
    main()
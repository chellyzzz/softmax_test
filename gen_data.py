import random
import numpy as np
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser(description="Generate input data for FP16 or FP32.")
parser.add_argument("--dtype", type=str, choices=["fp16", "fp32"], default="fp16",
                    help="Data type to generate: 'fp16' or 'fp32' (default: 'fp16').")
args = parser.parse_args()

N = 256
dtype = args.dtype

with open("src/include/input_data.h", "w") as f:
    f.write("#ifndef _INPUT_DATA_H_\n#define _INPUT_DATA_H_\n\n")
    f.write(f"#define ARRAY_SIZE {N}\n")
    f.write("#include <stdint.h>\n\n")

    if dtype == "fp16":
        f.write("uint16_t input_data[ARRAY_SIZE] = {\n")
        for i in range(N):
            value = random.uniform(-10, 10)
            fp16_value = np.float16(value)
            # 将 FP16 数值转为 uint16 的二进制表示
            uint16_value = fp16_value.view(np.uint16)
            f.write(f"    0x{uint16_value:04X},\n")
    else:  # fp32
        f.write("uint32_t input_data[ARRAY_SIZE] = {\n")
        for i in range(N):
            value = random.uniform(-10, 10)
            fp32_value = np.float32(value)
            # 将 FP32 数值转为 uint32 的二进制表示
            uint32_value = fp32_value.view(np.uint32)
            f.write(f"    0x{uint32_value:08X},\n")

    f.write("};\n\n#endif // _INPUT_DATA_H_\n")

print(f"Generated {N} {dtype.upper()} values in 'src/include/input_data.h'.")
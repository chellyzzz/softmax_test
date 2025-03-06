import random
import numpy as np

N = 2048
with open("src/include/input_data.h", "w") as f:
    f.write("#ifndef _INPUT_DATA_H_\n#define _INPUT_DATA_H_\n\n")
    f.write(f"#define ARRAY_SIZE {N}\n")
    f.write("#include <stdint.h>\n\n")
    f.write("uint16_t input_data[ARRAY_SIZE] = {\n")
    for i in range(N):
        value = random.uniform(-10, 10)
        fp16_value = np.float16(value)
        # 将FP16数值转为uint16的二进制表示
        uint16_value = fp16_value.view(np.uint16)
        f.write(f"    0x{uint16_value:04X},\n")
    f.write("};\n\n#endif // _INPUT_DATA_H_\n")

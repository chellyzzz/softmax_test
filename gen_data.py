import random

N = 4096
with open("src/include/input_data.h", "w") as f:
    f.write("#ifndef _INPUT_DATA_H_\n#define _INPUT_DATA_H_\n\n")
    f.write(f"#define ARRAY_SIZE {N}\n")
    f.write("float input_data[ARRAY_SIZE] = {\n")
    for i in range(N):
        value = random.uniform(-10, 10)
        f.write(f"    {value:.6f}f,\n")
    f.write("};\n\n#endif // _INPUT_DATA_H_\n")

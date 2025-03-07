CC = riscv64-unknown-elf-gcc
TOP = main

BUILD_DIR = build
TARGET = $(BUILD_DIR)/$(TOP)

INCLUDE_DIR = src/include
SRC += src/$(TOP).c
SRC += $(INCLUDE_DIR)/start.S


CFLAGS  += -O2 -march=rv64gcv_zvfh -mabi=lp64d -I$(INCLUDE_DIR)
CFLAGS  += -nostdlib  -fPIC
LDFLAGS += -T $(INCLUDE_DIR)/linker.ld

default: gen_data run disa

# 生成规则
run: $(SRC)
	make gen_data
	$(CC) $(CFLAGS) $(LDFLAGS) $(SRC) -o $(TARGET)

disa: $(TARGET)
	riscv64-unknown-elf-objdump -d $(TARGET) > $(TARGET).txt

clean:
	rm -f $(BUILD_DIR)/*

gen_data:
	python3 gen_data.py	--dtype fp32


.PHONY: clean run disa default	
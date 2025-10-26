JAVA_HOME ?= $(shell /usr/libexec/java_home)

GEMBED_DIR = ../gembed
GEMBED_TARGET = $(GEMBED_DIR)/target/release
GEMBED_LIB = $(GEMBED_TARGET)/libgembed.a

TARGET = libgembed_jni.dylib
SRC = jni_adapter.c
OBJS = jni_adapter.o

CFLAGS = -fPIC -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/darwin
LDFLAGS = \
	-lobjc -lc++ \
	-framework Foundation \
	-framework CoreML \
	-framework Security \
	-framework CoreFoundation

all: $(TARGET)

$(GEMBED_LIB):
	cd $(GEMBED_DIR) && cargo build --release

$(OBJS): $(SRC)
	clang $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJS) $(GEMBED_LIB)
	clang++ -shared -o $@ $(OBJS) $(GEMBED_LIB) $(LDFLAGS)

clean:
	rm -f $(OBJS) $(TARGET)
	cd $(GEMBED_DIR) && cargo clean

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Linux)
	JAVA_HOME ?= $(shell dirname $$(dirname $$(readlink -f $$(which javac))))
	OS_INCLUDES = -I$(JAVA_HOME)/include/linux
	LDFLAGS = -shared -Wl,--no-as-needed -ldl -lm -lstdc++ -lssl -lcrypto
	TARGET = libgembed_jni.so
	CXX = g++
	CC = gcc
else ifeq ($(UNAME_S),Darwin)
	JAVA_HOME ?= $(shell /usr/libexec/java_home)
	OS_INCLUDES = -I$(JAVA_HOME)/include/darwin
	LDFLAGS = -shared -lobjc -lc++ -framework Foundation -framework CoreML -framework Security -framework CoreFoundation
	TARGET = libgembed_jni.dylib
	CXX = clang++
	CC = clang
endif

GEMBED_DIR = ../gembed
GEMBED_TARGET = $(GEMBED_DIR)/target/release
GEMBED_LIB = $(GEMBED_TARGET)/libgembed.a

SRC = jni_adapter.c
OBJS = jni_adapter.o

CFLAGS = -fPIC -I$(JAVA_HOME)/include $(OS_INCLUDES)

all: $(TARGET)

$(GEMBED_LIB):
	cd $(GEMBED_DIR) && cargo build --release

$(OBJS): $(SRC)
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJS) $(GEMBED_LIB)
	$(CXX) -o $@ $(OBJS) $(GEMBED_LIB) $(LDFLAGS)

clean:
	rm -f $(OBJS) $(TARGET)
	cd $(GEMBED_DIR) && cargo clean

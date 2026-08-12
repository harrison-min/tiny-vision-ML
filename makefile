COMPILER = g++
CFLAGS =  -Wall -Iinclude -MMD
LDFLAGS = 
LIBS = 
TARGET = bin/main
SOURCE = src/main.cpp src/tensorMath.cpp src/testSuite.cpp src/layer.cpp

OBJECTS = $(patsubst src/%.cpp, obj/%.o, $(SOURCE))

all: dir $(TARGET)

$(TARGET): $(OBJECTS)
	$(COMPILER) $(CFLAGS) -o $@ $(OBJECTS) $(LDFLAGS) $(LIBS)

obj/%.o: src/%.cpp 
	$(COMPILER) $(CFLAGS) -c $< -o $@

dir:
	mkdir -p bin obj

clean:
	rm -rf bin obj
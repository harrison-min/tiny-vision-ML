COMPILER = g++
CFLAGS =  -Wall
LDFLAGS = 
LIBS = 
TARGET = bin/main
SOURCE = src/main.cpp src/tensorMath.cpp

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
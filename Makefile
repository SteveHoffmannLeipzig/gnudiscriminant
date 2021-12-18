CXX       := gcc
CXX_FLAGS := -g -I/usr/local/zlib/include/ -I/usr/include/

BIN     := bin
SRC     := src
INCLUDE := include

LIBRARIES   := -L/usr/lib/x86_64-linux-gnu/ -lgsl -lgslcblas -lm -lgmp -L/usr/local/zlib/lib -lz
EXECUTABLE  := gnulda


all: $(BIN)/$(EXECUTABLE)

run:clean all
	clear
	./$(BIN)/$(EXECUTABLE)

$(BIN)/$(EXECUTABLE): $(SRC)/*.c
	$(CXX) $(CXX_FLAGS) -I$(INCLUDE) $^ -o $@ $(LIBRARIES)

clean:
	-rm $(BIN)/main

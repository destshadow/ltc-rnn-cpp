CXX ?= g++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -Wpedantic
SOURCES := $(wildcard *.cpp)
HEADERS := $(wildcard *.hpp)
TARGET := main
TEST_TARGET := temporal_order_test
TEST_SOURCES := $(filter-out main.cpp,$(SOURCES)) tests/TemporalOrderTest.cpp
GENERAL_TEST_TARGET := generalization_test
GENERAL_TEST_SOURCES := $(filter-out main.cpp,$(SOURCES)) tests/general1.cpp

.PHONY: all run test clean

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

$(TEST_TARGET): $(TEST_SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(TEST_SOURCES) -o $(TEST_TARGET)

$(GENERAL_TEST_TARGET): $(GENERAL_TEST_SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(GENERAL_TEST_SOURCES) -o $(GENERAL_TEST_TARGET)

test: $(TEST_TARGET) $(GENERAL_TEST_TARGET)
	./$(TEST_TARGET)
	./$(GENERAL_TEST_TARGET)

clean:
	$(RM) $(TARGET) $(TEST_TARGET) $(GENERAL_TEST_TARGET)

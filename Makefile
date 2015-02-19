CXX = g++
CXXFLAGS = -O2 -g -Wall -std=c++0x

# Strict compiler options
CXXFLAGS += -Werror -Wformat-security -Wignored-qualifiers -Winit-self \
		-Wswitch-default -Wfloat-equal -Wshadow -Wpointer-arith \
		-Wtype-limits -Wempty-body -Wlogical-op \
		-Wmissing-field-initializers -Wctor-dtor-privacy \
		-Wnon-virtual-dtor -Wstrict-null-sentinel -Wold-style-cast \
		-Woverloaded-virtual -Wsign-promo -Wextra -pedantic -Wno-deprecated

# Directories with source code

# Directory, containing what to need to compile
SRC_DIR = vs2013
LIBLINEAR_DIR = $(SRC_DIR)/liblinear
ARGV_DIR = $(SRC_DIR)/argvparser
INCLUDE_DIR = /usr/local/include $(ARGV_DIR) $(LIBLINEAR_DIR)
LIB_DIR = /usr/local/lib

# Which libraries to prepare for usage: targets must be defined in BRIDGE_MAKE.
# easybmp replaced with opencv
BRIDGE_TARGETS = argvparser liblinear

# Link libraries gcc flag: library will be searched with prefix "lib".
OPENCV_FLAGS = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lopencv_ml
LDFLAGS = $(OPENCV_FLAGS)

# Add headers dirs to gcc search path
CXXFLAGS += -I $(INCLUDE_DIR)
# Add path with compiled libraries to gcc search path
CXXFLAGS += -L $(LIB_DIR)

# Helper macros
# subst is sensitive to leading spaces in arguments.
make_path = $(addsuffix $(1), $(basename $(subst $(2), $(3), $(4))))
# Takes path list with source files and returns pathes to related objects.
src_to_obj = $(call make_path,.o, $(SRC_DIR), $(OBJ_DIR), $(1))
# Takes path list with object files and returns pathes to related dep. file.
# Dependency files will be generated with gcc -MM.
src_to_dep = $(call make_path,.d, $(SRC_DIR), $(DEP_DIR), $(1))

# All source files in our project that must be built into movable object code.
CXXFILES := $(wildcard $(SRC_DIR)/*.cpp)
OBJFILES := $(call src_to_obj, $(CXXFILES))

# Default target (make without specified target).
.DEFAULT_GOAL := all

# Alias to make all targets.
.PHONY: all
all: $(CXXFILES) $(OBJFILES)
	$(CXX) $(CXXFLAGS) $(CXXFILES) $(OBJFILES) $(filter %.o, $^) -o $@ $(LDFLAGS)

# Suppress makefile rebuilding.
Makefile: ;

# If you still have "WTF?!" feeling, try reading teaching book
# by Mashechkin & Co. http://unicorn.ejudge.ru/instr.pdf

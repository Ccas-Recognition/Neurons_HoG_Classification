# This makefile is bridge between
# a library stuff and stuff that the library exports to world
# and, particularly, exports into our project.
# This makefile manages header and library files we use.
# Every file in these folders updates with owner Makefile.

INCLUDE_DIR = /usr/local/include
LIBRARY_DIR = /usr/local/lib

TARGET = argvparser
.DEFAULT_GOAL := $(TARGET)

# Avoid useless remaking of this Makefile.
Makefile : ;

# Liblinear library
# Path to Liblinear library
LIBLINEAR_LOCATION = ../liblinear
# Makefile name of Liblinear
LIBLINEAR_MAKEFILE = Makefile

# Path to files in the Liblinear directory to import in the main project
# Headers
LIBLINEAR_HEADERS = linear.h
# Statically compiled library (.a means archive)
LIBLINEAR_LIBRARY = liblinear.a

.PHONY: liblinear
# Copy prepared for import files from Liblinear directory
liblinear: $(LIBLINEAR_HEADERS) $(LIBLINEAR_LIBRARY)
	cp $(addprefix $(LIBLINEAR_LOCATION)/, $(LIBLINEAR_HEADERS)) $(INCLUDE_DIR)/
	cp $(addprefix $(LIBLINEAR_LOCATION)/, $(LIBLINEAR_LIBRARY)) $(LIBRARY_DIR)/

.PHONY: $(LIBLINEAR_HEADERS)
# Prepare library headers by calling to Liblinear makefile
$(LIBLINEAR_HEADERS):
	make -C $(LIBLINEAR_LOCATION) -f $(LIBLINEAR_MAKEFILE) $@

.PHONY: $(LIBLINEAR_LIBRARY)
# Prepare library by calling to Liblinear makefile
$(LIBLINEAR_LIBRARY):
	make -C $(LIBLINEAR_LOCATION) -f $(LIBLINEAR_MAKEFILE) $@

# Argvparser library
# Path to Argvparser library
ARGVPARSER_LOCATION = ../argvparser
# Makefile name of Argvparser
ARGVPARSER_MAKEFILE = Makefile

# Path to files in the Argvparser directory to import in the main project
# Headers
ARGVPARSER_HEADERS = include/argvparser.h
# Statically compiled library (.a means archive)
ARGVPARSER_LIBRARY = lib/libargvparser.a

.PHONY: argvparser
# Copy prepared for import files from Argvparser directory
argvparser: $(ARGVPARSER_HEADERS) $(ARGVPARSER_LIBRARY)
	cp $(addprefix $(ARGVPARSER_LOCATION)/, $(ARGVPARSER_HEADERS)) $(INCLUDE_DIR)/
	cp $(addprefix $(ARGVPARSER_LOCATION)/, $(ARGVPARSER_LIBRARY)) $(LIBRARY_DIR)/

.PHONY: $(ARGVPARSER_HEADERS)
# Prepare library headers by calling to Argvparser makefile
$(ARGVPARSER_HEADERS):
	make -C $(ARGVPARSER_LOCATION) -f $(ARGVPARSER_MAKEFILE) $@

.PHONY: $(ARGVPARSER_LIBRARY)
# Prepare library by calling to Argvparser makefile
$(ARGVPARSER_LIBRARY):
	make -C $(ARGVPARSER_LOCATION) -f $(ARGVPARSER_MAKEFILE) $@

.PHONY: clean
clean:
	rm -f $(wildcard $(INCLUDE_DIR)/*) $(wildcard $(LIBRARY_DIR)/*)
	make -C $(ARGVPARSER_LOCATION) -f $(ARGVPARSER_MAKEFILE) $@
	make -C $(LIBLINEAR_LOCATION) -f $(LIBLINEAR_MAKEFILE) $@

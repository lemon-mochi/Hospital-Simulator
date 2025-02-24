CC=cc
CXX=CC
CCFLAGS= -g -std=c99 -Wall -Werror

# List of all source files
SRCS = hospital_simulator.c

# Generate object file names from source file names
OBJS = $(SRCS:.c=.o)

# Main target
all: hw6

# Rule to compile source files into object files
%.o : %.c
	$(CC) -c $(CCFLAGS) $<

# Rule to link object files into executable
hospital_simulator: $(OBJS)
	$(CC) -o hospital_simulator $(OBJS) -lm

# Clean rule to remove object files and executable
clean:
	rm -f *.o hospital_simulator
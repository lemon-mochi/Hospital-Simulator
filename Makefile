CC=cc
CXX=CC
CCFLAGS= -g -std=c99 -Wall -Werror

# List of all source files
SRCS = hw6.c

# Generate object file names from source file names
OBJS = $(SRCS:.c=.o)

# Main target
all: hw6

# Rule to compile source files into object files
%.o : %.c
	$(CC) -c $(CCFLAGS) $<

# Rule to link object files into executable
hw6: $(OBJS)
	$(CC) -o hw6 $(OBJS) -lm

# Clean rule to remove object files and executable
clean:
	rm -f *.o hw6
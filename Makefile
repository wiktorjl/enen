# Compiler and flags
CC = gcc
CFLAGS = -Wall
LDFLAGS = -lm

# All executables
all: xor gym accuracy

# Executable rules
xor: xor.o nn.o tools.o config.o
	$(CC) $(CFLAGS) -o xor xor.o nn.o tools.o config.o $(LDFLAGS)

gym: gym.o nn.o tools.o config.o
	$(CC) $(CFLAGS) -o gym gym.o nn.o tools.o config.o $(LDFLAGS)

accuracy: accuracy.o nn.o tools.o config.o
	$(CC) $(CFLAGS) -o accuracy accuracy.o nn.o tools.o config.o $(LDFLAGS)

config: config.o tools.o
	$(CC) $(CFLAGS) -o config config.o tools.o $(LDFLAGS)

# Object file rules
xor.o: xor.c nn.h
	$(CC) $(CFLAGS) -c xor.c

gym.o: gym.c nn.h
	$(CC) $(CFLAGS) -c gym.c

accuracy.o: accuracy.c nn.h
	$(CC) $(CFLAGS) -c accuracy.c

nn.o: nn.c nn.h tools.h
	$(CC) $(CFLAGS) -c nn.c

tools.o: tools.c tools.h
	$(CC) $(CFLAGS) -c tools.c

config.o: config.c config.h tools.h
	$(CC) $(CFLAGS) -c config.c


# Clean rule
clean:
	rm -f xor gym accuracy config *.o
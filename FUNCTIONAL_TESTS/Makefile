
CXXFLAGS=-O2 -std=c++11 
#CXXFLAGS+=-fopenmp
CFLAGS+=-O2 -std=c99

all: hbfarm425 checkramp checkramp480 multisitecheckramp checkES isramp

clean:
	rm -f hbfarm425 checkramp


multisitecheckramp:
	g++ $(CXXFLAGS) multisitecheckramp.cpp -o multisitecheckramp -lpopt

myinstall:
	cp hbfarm425 checkramp checkramp480 multisitecheckramp ~/bin

install:
	cp hbfarm425 checkramp /usr/local/bin
	
